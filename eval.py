import torch
import tqdm
import os.path
import json
from PIL import Image
import torch.nn.functional as F


class ImageEmbedder:
    def __init__(self, model, preprocessor):
        """ model projects image to vector, processor load and prepare image to the model"""
        self.model = model
        self.processor = preprocessor


class Corpus(torch.utils.data.Dataset):
    """ Dataset class for the corpus images (the 50k potential candidates)"""
    def __init__(self, corpus_path, preprocessor):
        with open(corpus_path) as f:
            self.corpus = json.load(f)
        self.preprocessor = preprocessor
        self.path2id = {self.corpus[i]: i for i in range(len(self.corpus))}

    def __len__(self):
        return len(self.corpus)

    def path_to_index(self, path):
        """ For finding a target image fast"""
        return self.path2id[path]

    def __getitem__(self, i):
        image = self.preprocessor(self.corpus[i])  # Load and prepare image
        return {'id': i, 'image': image}


class Queries(torch.utils.data.Dataset):
    """ Dataset class for the queries and their targets (dialog and image)"""
    def __init__(self, cfg, queries_path):
        with open(queries_path) as f:
            self.queries = json.load(f)

        self.dialog_length = None  # Set the dialog length to evaluate on
        self.cfg = cfg

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, i):
        assert self.dialog_length is not None, "Please set self.dialog_length=<DIALOG_LENGTH> to any number [0,..,10]"
        target_path = self.queries[i]['img']
        # Concatenate the partial dialog information with a predefined seperator.
        text = self.cfg['sep_token'].join(self.queries[i]['dialog'][:self.dialog_length + 1])
        return {'text': text, 'target_path': target_path}


class ChatIREval:
    """ This class run the main evaluation process.
    """
    def __init__(self, cfg, dialog_encoder, image_embedder: ImageEmbedder):
        self.dialog_encoder = dialog_encoder  # In paper was referred as "Image Retriever"
        self.image_embedder = image_embedder  # Image encoder

        self.cfg = cfg
        self.corpus = None
        self.corpus_dataset = Corpus(self.cfg['corpus_path'], self.image_embedder.processor)

    def _get_recalls(self, dataloader, dialog_length):
        # Set dialog length
        dataloader.dataset.dialog_length = dialog_length
        recalls = []
        for batch in tqdm.tqdm(dataloader):
            target_ids = torch.tensor([self.corpus_dataset.path_to_index(p) for p in batch['target_path']]).unsqueeze(1).to(self.cfg['device'])
            pred_vec = F.normalize(self.dialog_encoder(batch['text']), dim=-1)
            # batch recalls
            scores = pred_vec @ self.corpus[1].T
            arg_ranks = torch.argsort(scores, descending=True, dim=1).long()
            target_recall = ((arg_ranks - target_ids) == 0).nonzero()[:, 1]
            recalls.append(target_recall)

        return torch.cat(recalls)

    def run(self, hits_at=10):
        assert self.corpus, f"Prepare corpus first (self.index_corpus())"
        dataset = Queries(cfg, self.cfg['queries_path'])
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.cfg['queries_bs'],
                                                 shuffle=False,
                                                 num_workers=self.cfg['num_workers'],
                                                 pin_memory=True,
                                                 drop_last=False
                                                 )
        hits_results = []
        for dl in range(11):
            print(f"Calculate recalls for each dialogues of length {dl}...")
            dialog_recalls = self._get_recalls(dataloader, dialog_length=dl)
            hits_results.append(dialog_recalls)

        hits_results = cumulative_hits_per_round(torch.cat(hits_results).cpu(), hitting_recall=10).tolist()
        print("====== Results for Hits@10 ====== ")
        for dl in range(11):
            print(f"\t Dialog Length: {dl}: {round(hits_results[dl], 2)}%")

    def index_corpus(self):
        """ Prepare corpus (image search space)"""
        # self.corpus = torch.arange(50000).to(cfg['device']), torch.randn(50_000, 512).to(cfg['device']).half()
        if self.cfg['cache_corpus'] and os.path.exists(self.cfg['cache_corpus']):
            print(f"<<<<Cached corpus has been loaded: {self.cfg['cache_corpus']} >>>>>")
            print(f"Warning: Make sure this corpus has been indexed with the right image embedder!")
            self.corpus = torch.load(self.cfg['cache_corpus'])
            return
        # return
        dataloader = torch.utils.data.DataLoader(self.corpus_dataset,
                                                 batch_size=self.cfg['corpus_bs'],
                                                 shuffle=False,
                                                 num_workers=self.cfg['num_workers'],
                                                 pin_memory=True,
                                                 drop_last=False
                                                 )
        print("Preparing corpus (search space)...")
        corpus_vectors = []
        corpus_ids = []
        for batch in tqdm.tqdm(dataloader):
            batch_vectors = F.normalize(self.image_embedder.model(batch['image'].to(self.cfg['device'])), dim=-1)
            corpus_vectors.append(batch_vectors)
            corpus_ids.append(batch['id'].to(self.cfg['device']))

        corpus_vectors = torch.cat(corpus_vectors)
        corpus_ids = torch.cat(corpus_ids)

        # sort by id: important!
        arg_ids = torch.argsort(corpus_ids)
        corpus_vectors = corpus_vectors[arg_ids]
        corpus_ids = corpus_ids[arg_ids]

        self.corpus = corpus_ids, corpus_vectors
        if self.cfg['cache_corpus']:
            torch.save(self.corpus, self.cfg['cache_corpus'])


def get_first_hitting_time(target_recall, hitting_recall=10):
    """ returns (11, n) tensor with hitting time in each round (0, 11). inf indicate a miss (no hit after 11 rounds) """
    target_recalls = target_recall.view(11, -1).T
    hits = (target_recalls < hitting_recall)

    final_hits = torch.inf * torch.ones(target_recalls.shape[0])

    hitting_times = []
    for ro_i in range(11):
        rh = hits[:, ro_i]
        final_hits[rh] = torch.min(final_hits[rh], torch.ones(final_hits[rh].shape) * ro_i)
        hitting_times.append(final_hits.clone())

    return torch.stack(hitting_times)


def cumulative_hits_per_round(target_recall, hitting_recall=10):
    """ return calculation of avg number of hits until round x"""
    if type(hitting_recall) is tuple:
        assert len(hitting_recall) == 1
        hitting_recall = hitting_recall[0]
    ht_times = get_first_hitting_time(target_recall, hitting_recall)
    return ((ht_times < torch.inf).sum(dim=-1) * 100 / ht_times[0].shape[0])


def CLIP_ZERO_SHOT_BASELINE():
    # Install CLIP library from https://github.com/openai/CLIP
    import clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("ViT-B/32", device='cpu')
    model, preprocess = clip.load("ViT-B/16", device='cpu')
    model = model.to(device)
    image_embedder = ImageEmbedder(lambda img: model.encode_image(img), lambda path: preprocess(Image.open(path)))
    # Note that CLIP supports only 77 tokens!! this is just a baseline.
    dialog_encoder = lambda text: model.encode_text(clip.tokenize(text, truncate=True).to(device))

    return dialog_encoder, image_embedder


if __name__ == '__main__':

    cfg = {'corpus_bs': 500,
           'queries_bs': 500,
           'num_workers': 8,
           'sep_token': ', ',  # Separation between dialog rounds
           'cache_corpus': "temp/corpus_clip_16.pth",  # Cache path for saving indexed corpus
            'queries_path': 'dialogues/VisDial_v1.0_queries_val.json',
            'corpus_path': 'ChatIR_Protocol/Search_Space_val_50k.json',
            'device': 'cuda:0',  # 'cpu'
           }

    with torch.no_grad():
        dialog_encoder, image_embedder = CLIP_ZERO_SHOT_BASELINE()
        evaluator = ChatIREval(cfg, dialog_encoder, image_embedder)
        evaluator.index_corpus()
        evaluator.run(hits_at=10)  # Hit@10 as in the paper

