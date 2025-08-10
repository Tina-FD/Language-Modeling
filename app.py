import streamlit as st
import torch
from torch import nn
from torchtext.data.utils import get_tokenizer

class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        return

    def _setup(self):
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition
            for name_w in self.weights:
                w = getattr(self.module, name_w)
                del self.module._parameters[name_w]
                self.module.register_parameter(name_w + '_raw', nn.Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            mask = torch.nn.functional.dropout(torch.ones_like(raw_w), p=self.dropout, training=True) * (1 - self.dropout)
            setattr(self.module, name_w, raw_w * mask)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)

def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    return torch.nn.functional.embedding(words, masked_embed_weight, padding_idx, embed.max_norm, embed.norm_type, embed.scale_grad_by_freq, embed.sparse)

class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout, self).__init__()

    def forward(self, x, dropout):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = m.requires_grad_(False) / (1 - dropout)
        return mask.expand_as(x) * x

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropoute=0.2, dropouti=0.2, dropouth=0.2, dropouto=0.2, weight_drop=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.uniform_(-0.1, 0.1)

        self.lstms = [nn.LSTM(embedding_dim, hidden_dim, num_layers=1, dropout=0, batch_first=False) for _ in range(num_layers)]
        if weight_drop > 0:
            self.lstms = [WeightDrop(lstm, ['weight_hh_l0'], dropout=weight_drop) for lstm in self.lstms]
        self.lstms = nn.ModuleList(self.lstms)

        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.fc.weight = self.embedding.weight

        self.lockdrop = LockedDropout()
        self.dropoute = dropoute
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropouto = dropouto

    def forward(self, src):
        embedding = embedded_dropout(self.embedding, src, dropout=self.dropoute if self.training else 0)
        embedding = self.lockdrop(embedding, self.dropouti)

        for l, lstm in enumerate(self.lstms):
            embedding, _ = lstm(embedding)
            if l != self.num_layers - 1:
                embedding = self.lockdrop(embedding, self.dropouth)

        return self.fc(self.lockdrop(embedding, self.dropouto))

embedding_dim = 300
num_layers = 3
hidden_dim = 1150
dropoute = 0.1
dropouti = 0.65
dropouth = 0.3
dropouto = 0.4
weight_drop = 0.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load('model.pt', map_location=torch.device(device))
model.eval()
tokenizer = get_tokenizer('basic_english')
vocab = torch.load('Wikitext_train_vocab.pt')

LanguageModel(len(vocab), 300, 512, 2)

def generate(prompt, tokenizer=tokenizer, vocab=vocab, model=model, max_seq_len=6, temperature=0.5, num_pred=4, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    itos = vocab.get_itos()
    preds = []
    for _ in range(num_pred):
        seq = prompt
        indices = vocab(tokenizer(seq))
        for i in range(max_seq_len):
            src = torch.LongTensor(indices).to(device)
            with torch.no_grad():
                prediction = model(src)

            probs = torch.softmax(prediction[-1]/temperature, dim=0)
            idx = vocab['<ukn>']
            while idx == vocab['<ukn>']:
                idx = torch.multinomial(probs, num_samples=1).item()

            token = itos[idx]
            seq += ' ' + token
            if idx == vocab['.']:
                break

            indices.append(idx)
        preds.append(seq)

    return preds

st.title("Language Modeling")

background_style = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-color: #48cae4;
}
h1 {
    color: #023e8a;
    font-size: 36px;
    margin-bottom: 10px;
    text-align: center;
}
</style>
"""

st.markdown(background_style, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .search-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 30vh;
        flex-direction: column;
        width: 60%;
        margin-top: 10px;
    }
    .suggestion {
        background-color: #eaeaea;
        padding: 8px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        position: relative;
        font-size: 16px;
        color: #555;
        width: 95%;
        margin-top: 8px;
    }
    .search-bar {
        width: 50%;
        position: relative;
    }
    .user-input {
        width: 100%;
        padding: 12px;
        font-size: 18px;
        border: 2px solid #0077b6;
        border-radius: 25px;
        box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.2);
        color: #03045e;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ""

st.markdown('<div class="search-container">', unsafe_allow_html=True)
user_input = st.text_input("", placeholder="write something ...", label_visibility="collapsed", key="input_box")

if user_input != st.session_state['user_input']:
    st.session_state['user_input'] = user_input

if st.session_state['user_input']:
    suggestions = generate(st.session_state['user_input'])
    for suggestion in suggestions:
        st.markdown(f'<div class="suggestion">{suggestion}</div>', unsafe_allow_html=True)

st.sidebar.title("Project details")
st.sidebar.markdown("""
<style>
    .sidebar .sidebar-content {
        background-color: #023e8a;
        color: white;
    }
    a {
        color: #00b4d8;
    }
</style>
""", unsafe_allow_html=True)

project_name = "Language Model with Lstm "
project_link = "https://github.com/shgyg99/LanguageModeling/blob/main/app.py"

if project_name and project_link:
    st.sidebar.markdown(f"- [{project_name}]({project_link})")

if project_name and project_link:
    st.sidebar.write(f"{project_name}]({project_link})")
