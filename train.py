from utils import *
from config import *
from model import *
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer
import re

import numpy as np



train_path="./data/train.jsonl"
test_path="./data/test.jsonl"



load_new_data=False
laod_new_encode=False

def load_new():
    train=load_json_asDataFrame(train_path)
    test=load_json_asDataFrame(train_path)
    train.to_csv("./data/train.csv", mode='w')
    test.to_csv("./data/test.csv", mode='w')

def target_init(y):
    #########first init######
    parse=re.sub('[,]', '', y[1][1:-1]).split()
    print(parse)
    out= np.array((list(map(int, parse)))).reshape(1,3)
    #########################

    for i in y:

        parse = re.sub('[,]', '', i[1:-1]).split()
        parse = list(map(int, parse))
        print(i)
        parse=np.array((parse)).reshape(1,3)
        out = np.append(out,parse,axis=0)


    return out

def Tokenizer(item):
    item=list(np.array(item.tolist()))
    max=0
    tok_path = get_tokenizer()
    model, vocab = get_pytorch_kogpt2_model()
    tok = SentencepieceTokenizer(tok_path, num_best=0, alpha=0)

    print(item[6])

    ################################init first tensor##################################

    toked = tok(item[0])
    input_ids = torch.tensor([vocab[vocab.bos_token], ] + vocab[toked]).unsqueeze(0)
    size = input_ids.shape
    out=torch.cat([input_ids, torch.empty(1, max_seqlen-size[1])], axis=1)

    ###################################################################################

    for i in item:

        toked = tok(i)
        input_ids = torch.tensor([vocab[vocab.bos_token], ] + vocab[toked]).unsqueeze(0)
        size=input_ids.shape
        # print(input_ids)
        # print(input_ids.shape)
        y=torch.cat([input_ids, torch.empty(1, max_seqlen-size[1])], axis=1)
        out=torch.cat([out,y],axis=0)

        print(out.shape)


    x_np = out.numpy()
    x_df = pd.DataFrame(x_np)
    x_df.to_csv('./data/encoded.csv', mode='w')

def data_processing(train):
    x_media=train['media'].copy()
    x_id=train['id'].copy()
    x_article=train['article_original'].copy()

    y=train['extractive'].copy()
    x = train.copy().drop(['extractive'],axis='columns', inplace=True)

    return x_id,x_media,x_article,y,x

class Trainer(object):
    def __init__(self):

        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.reduce_factor = reduce_factor
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.model = None


    def train(self):
        if(load_new_data):
            load_new()

        low_train=pd.read_csv("./data/train.csv")
        low_test=pd.read_csv("./data/test.csv")

        x_media,x_id,x_article,y,x=data_processing(low_train)
        x_article=x_article.to_numpy()
        y=target_init(y)
        y=torch.tensor(y)


        if(laod_new_encode):
            Tokenizer(x_article)

        x_article = pd.read_csv("./data/encoded.csv")
        x_article=torch.tensor(x_article.to_numpy())
        print(x_article.shape)
        print(y.shape)
        dataset=torch.utils.data.TensorDataset(x_article, y)


        dataloader=train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.patience,factor=self.reduce_factor)



        for i_epoch in range(n_epoch):
            self.model.train()
            losses = 0
            size = 0
            train_diff = 0
            traan_total = 0

            for _, (x_mb, y_mb) in enumerate(train_dataloader):
                x_mb = x_mb.to(self.device)
                y_mb = y_mb.to(self.device)

                self.model.zero_grad()
                optimizer.zero_grad()
                y_mb = y_mb.squeeze(2)

                y_hat_mb = self.model(x_mb)
                loss = criterion(y_hat_mb, y_mb)
                loss.backward()

                optimizer.step()
                losses += loss.item()
                size += 1
                train_diff += self.get_MAPE(y_mb, y_hat_mb)

                traan_total += len(y_mb) * self.seq_len

            scheduler.step(loss)
            train_score = (train_diff / traan_total) * 100

            loss_avg = losses / size
            lr = self.get_lr(optimizer)

            print(f'epoch: {i_epoch} \tTrainLoss: {loss_avg}\t MAPE: {train_score} \tlr: {lr} ', end="\n\n")

            # save model
            # filename = 'model_' + str(i_epoch) + '.pth'
            # torch.save(self.model, './saves/' + filename)
        print("??")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
    print("training done")
