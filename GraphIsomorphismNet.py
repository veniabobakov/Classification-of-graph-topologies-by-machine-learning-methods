from torch import ones
from torch_geometric import nn as pyg_nn
from torch import nn 
from torch.nn import functional as F


class GNNStack(nn.Module):  # класс нашей сети
    """"
        input_dim - количество измерений в признаке вершины
        hidden_dim - количество скрытых слоев
        output_dim - количество выходных параметров
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers = 3):
        super(GNNStack, self).__init__()  # наследуем nn.Module
        self.num_layers = num_layers
        self.convs = nn.ModuleList()  # создаем пустой список слоев для сверток
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))  # вставляем входной сверточный слой
        self.lns = nn.ModuleList()  # создаем списов линейных слоев
        self.lns.append(nn.LayerNorm(hidden_dim))  # вставляем 2 линейных слоя
        self.lns.append(nn.LayerNorm(hidden_dim))
        for _ in range(num_layers - 1):  # вставляем скрытые сверточные слои
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        self.post_mp = nn.Sequential(  # подаем результаты сверток на поолносвязную сеть
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25),
            nn.Linear(hidden_dim, output_dim))

        self.dropout = 0.25  # вероятность для дропаута

    def build_conv_model(self, input_dim, hidden_dim):  # функция для создания сверточного слоя для регрессии графов
        return pyg_nn.GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                            nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))

    """"
    В этой реализации программы будем использовать Graph Isomorphism Network(GIN) сверточные слои из статьи
    "How Powerful are Graph Neural Networks?" Keyulu Xu

    """

    def forward(self, data):  # функция прямого распространения
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if data.num_node_features == 0:
            x = ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            emb = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        x = pyg_nn.global_mean_pool(x, batch)  # усредняем значения для графа

        x = self.post_mp(x)

        return emb, F.log_softmax(x, dim=1)
