import torch
from torch import nn
from torch.nn import functional as f
from torch.optim import Adam
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
import gc
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import scikitplot as skplt
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = get_tokenizer("basic_english")


def load_data():
    df = pd.read_csv('data/data.csv')
    x = df["Text"].values
    y = df["Tag"].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return x_train, x_test, y_train, y_test


def build_vocab(datasets):
    for dataset in datasets:
        for sentence in dataset:
            yield tokenizer(sentence)


target_classes = ["HR", "Legal", "Finance"]
x_train, x_test, y_train, y_test = load_data()
vocab = build_vocab_from_iterator(build_vocab([x_train, x_test]), specials=["<UNK>"])
vocab.set_default_index(vocab["<UNK>"])

vectorizer = CountVectorizer(vocabulary=vocab.get_itos(), tokenizer=tokenizer)


def vectorize_batch(batch):  # Needs to be passed a tuple ((tag, text))
    y, x = list(zip(*batch))
    x = vectorizer.transform(x).todense()
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y)


class CreateDataset(Dataset):
    def __init__(self, y, x):
        self.y = y
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.y[idx], self.x[idx]


class CreatePredictDataset(Dataset):
    def __init__(self, x):
        self.x = x
        self.y = [0]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.y[0], self.x[idx]


train_ds = CreateDataset(y_train, x_train)
test_ds = CreateDataset(y_test, x_test)

train_loader = DataLoader(train_ds, batch_size=256, collate_fn=vectorize_batch)
test_loader = DataLoader(test_ds, batch_size=256, collate_fn=vectorize_batch)


class TextClassifier(nn.Module):
    def __init__(self):
        super(TextClassifier, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(len(vocab), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x_batch):
        return self.seq(x_batch)


text_classifier = TextClassifier()


def loss_and_acc_calc(model, loss_fn, val_loader):
    with torch.no_grad():
        y_shuffled, y_pred, losses = [], [], []
        for X, Y in val_loader:
            pred = model(X)
            loss = loss_fn(pred, Y)
            losses.append(loss.item())
            y_shuffled.append(Y)
            y_pred.append(pred.argmax(dim=-1))

        y_shuffled = torch.cat(y_shuffled)
        y_pred = torch.cat(y_pred)

        print("Valid Loss : {:.3f}".format(torch.tensor(losses).mean()))
        print("Valid Acc  : {:.3f}".format(accuracy_score(y_shuffled.detach().numpy(), y_pred.detach().numpy())))


def train_model(model, loss_fn, optimizer, train_loader, val_loader, epochs=10):
    for i in range(1, epochs+1):
        losses = []
        for X, Y in tqdm(train_loader):
            y_preds = model(X)

            loss = loss_fn(y_preds, Y)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()#

        print("Train Loss : {:.3f}".format(torch.tensor(losses).mean()))
        loss_and_acc_calc(model, loss_fn, val_loader)


epochs = 25
learning_rate = 1e-4

loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(text_classifier.parameters(), lr=learning_rate)

train_model(text_classifier, loss_fn, optimizer, train_loader, test_loader, epochs)


def make_predictions(model, loader):
    y_shuffled, y_preds = [], []
    for X, Y in loader:
        preds = model(X)
        y_preds.append(preds)
        y_shuffled.append(Y)
    gc.collect()
    y_preds, y_shuffled = torch.cat(y_preds), torch.cat(y_shuffled)

    return y_shuffled.detach().numpy(), f.softmax(y_preds, dim=-1).argmax(dim=-1).detach().numpy()


Y_actual, Y_preds = make_predictions(text_classifier, test_loader)

print("Test Accuracy : {}".format(accuracy_score(Y_actual, Y_preds)))
print("\nClassification Report : ")
print(classification_report(Y_actual, Y_preds, target_names=target_classes))
# skplt.metrics.plot_confusion_matrix([target_classes[i] for i in Y_actual], [target_classes[i] for i in Y_preds],
#                                     normalize=True,
#                                     title="Confusion Matrix",
#                                     cmap="Blues",
#                                     hide_zeros=True,
#                                     figsize=(5,5)
#                                     )
# plt.xticks(rotation=90)
# plt.show()


def new_prediction(text, model):
    vocab = build_vocab_from_iterator(build_vocab(text), specials=["<UNK>"])
    vocab.set_default_index(vocab["<UNK>"])
    predict_ds = CreatePredictDataset(text)
    predict_loader = DataLoader(predict_ds, batch_size=1, collate_fn=vectorize_batch)
    for X, Y in tqdm(predict_loader):
        y_preds = model(X)
    print(f"Prediction: {target_classes[torch.argmax(y_preds)]}\n\n{y_preds}")


new_pred_text = """To file harassment. Hi, I’ll try to make this brief. I work in a highly regulated industry. I have a
boss who head hunted me based on our previous working relationship. Things were going well with us until the snapping
started. Then the passive aggressive emails and snide comments. The office is walking on pins and needles based on the
emotional outburst. Many documented “bad toned” emails, to myself and other staff. Including making staff cry. The last
year, I discovered fraudulent charges. However, there was an excuse that was accepted. Now, this has escalated as him
and I had an incident where I had to defend myself and my team. He spiraled out of control to attacking other parts of
my job and doing the “cc” himself tactic. I finally stood up to him and it’s been terrible ever since. He’s gone to his
boss to complain which was dismissed. He tried again, recently. Yelled at me in the office in front of others. My
question is, is it worth it? After an investigation does HR really side with the subordinate if it’s discovered? Like he
has lots of experience in our field. I just feel like I won’t have a leg to stand on due the power and experience. This 
is affecting my life tremendously
"""
new_prediction(new_pred_text, text_classifier)

