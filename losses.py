import torch
import torch.nn as nn
import torch.nn.functional as F
import model_params


class SemanticLoss(nn.Module):
    """Semantic loss.

      Reference:
      Sandareka et al. Comprehensible Convolutional Neural Network via Guided Concept Learning. IJCNN 2021.

      Args:
          num_concepts (int): number of concepts.
          out_dim (int): output dimension of the feature extractor, i.e. (N, C, H, W) -> out_dim = H*W
          emb_dim (int): embedding space dimension.
          text_dim (int): text space dimension.
          word_vectors (tensor): word vector representation of the class names.
          alpha (float): hyperparameter.
          device (torch.device): cuda or cpu.
    """

    def __init__(self, num_concepts, out_dim, emb_dim, text_dim, word_vectors, device, alpha=1):
        super(SemanticLoss, self).__init__()
        self.num_concepts = num_concepts
        self.out_dim = out_dim
        self.emb_dim = emb_dim
        self.text_dim = text_dim
        self.word_vectors = word_vectors
        self.device = device
        self.alpha = alpha

        # Define the learnable variables
        self.v_e_converter = nn.Parameter(torch.randn(self.out_dim, self.emb_dim))
        self.w_e_converter = nn.Parameter(torch.randn(self.text_dim, self.emb_dim))

    def forward(self, visual_feats, indicator_vectors):
        """
        Args:
            visual_feats: feature maps with shape (batch_size, num_concepts, 14, 14).
            indicator_vectors: ground truth indicator vectors with shape (batch_size, num_concepts).
        """
        # print(f"V_E_Converter: {self.v_e_converter}")
        # print(f"W_E_Converter: {self.w_e_converter}")

        # Reshape visual feats
        visual_feats = torch.reshape(
            torch.transpose(torch.reshape(visual_feats, [-1, self.num_concepts, self.out_dim]), 1, 2),
            [-1, self.out_dim])

        # Compute embedding vectors
        embedded_visual_feats = torch.matmul(visual_feats, self.v_e_converter.to(self.device))
        embedded_text_feats = torch.matmul(self.word_vectors.to(self.device), self.w_e_converter.to(self.device))

        # Normalize embedding vectors
        normalized_emd_visual_feats = F.normalize(embedded_visual_feats, p=2, dim=1)
        normalized_text_feats = F.normalize(embedded_text_feats, p=2, dim=1)

        # Compute Semantic Loss
        # shape = (96, 6)
        cosine_similarity = torch.matmul(normalized_emd_visual_feats, torch.transpose(normalized_text_feats, 0, 1))
        # shape = (16, 6, 6) -> 1 da Loss s
        cos_sim_reshaped = torch.reshape(cosine_similarity, [-1, self.num_concepts, self.num_concepts])
        # shape = (16, 6)
        positive = torch.sum(torch.multiply(cos_sim_reshaped, torch.eye(self.num_concepts).to(self.device)), dim=2)
        # shape = (16, 6, 6) -> 2 da Loss s
        metric_p = torch.tile(torch.unsqueeze(positive, dim=2), [1, 1, self.num_concepts])
        # shape = (16, 6, 6)
        delta = torch.subtract(cos_sim_reshaped, metric_p)
        # shape = (16, 6)
        semantic_loss = torch.multiply(
            torch.sum(F.relu(torch.add((self.alpha + delta), -2 * torch.eye(self.num_concepts).to(self.device))),
                      dim=2),
            indicator_vectors)

        return semantic_loss, positive


class CounterLoss(nn.Module):
    """Counter loss.

      Reference:
      Sandareka et al. Comprehensible Convolutional Neural Network via Guided Concept Learning. IJCNN 2021.

      Args:
          beta (float): hyperparameter.
    """

    def __init__(self, beta, device):
        super(CounterLoss, self).__init__()
        self.beta = beta
        self.device = device

    def forward(self, indicator_vectors, positive):
        """
        Args:
            indicator_vectors: ground truth indicator vectors with shape (batch_size, num_concepts).
            positive: matrix with shape (batch_size, num_concepts) denoting the ...
        """
        # ---------------------     Counter Loss    ---------------------
        indices = torch.arange(start=0, end=positive.shape[0], dtype=torch.int32)

        shuffled_indices = torch.randperm(positive.shape[0]).to(self.device)

        shuffled_positive = torch.gather(positive, 0, shuffled_indices.unsqueeze(dim=-1))

        shuffled_indicator_vectors = torch.gather(indicator_vectors, 0, shuffled_indices.unsqueeze(dim=-1))

        im = torch.subtract(indicator_vectors, shuffled_indicator_vectors)

        img_only_feats = F.relu(torch.subtract(indicator_vectors, shuffled_indicator_vectors))

        count_loss = torch.multiply(
            F.relu(
                torch.subtract(
                    torch.multiply(shuffled_positive, img_only_feats),
                    torch.multiply(positive, img_only_feats)
                ) + self.beta),
            img_only_feats)

        return count_loss


class UniquenessLoss(nn.Module):
    """Uniqueness loss.

      Reference:
      Sandareka et al. Comprehensible Convolutional Neural Network via Guided Concept Learning. IJCNN 2021.

      Args:
          num_concepts (int): number of concepts.
    """

    def __init__(self, num_concepts, device):
        super(UniquenessLoss, self).__init__()
        self.num_concepts = num_concepts
        self.device = device

    def forward(self, net, indicator_vectors):
        """
        Args:
            indicator_vectors: ground truth indicator vectors with shape (batch_size, num_concepts).
            net: matrix with shape (batch_size, num_concepts) denoting the output of GAP.
        """
        # ---------------------   Uniqueness Loss   ---------------------
        # zero-mean normalization - subtrair a mean activation da GAP
        sigmoid_inputs = torch.subtract(net, torch.tile(torch.unsqueeze(torch.mean(net, dim=1), dim=1),
                                                        [1, self.num_concepts]))

        #sigmoid_inputs = torch.sigmoid(sigmoid_inputs)
        sigmoid_inputs = torch.tanh(sigmoid_inputs)

        criterion = torch.nn.CrossEntropyLoss()
        uniqueness_loss = criterion(sigmoid_inputs + torch.tensor(1e-5), indicator_vectors)

        return uniqueness_loss


class ConceptLoss(nn.Module):
    """Concept Loss

        Calculates the MSE between a given segmentation mask and the output of the concept layer.
    """

    def __init__(self, device):
        super(ConceptLoss, self).__init__()
        self.device = device

    def forward(self, masks, concept_layer):
        criteria = torch.nn.MSELoss()
        loss = criteria(concept_layer.to(self.device), masks.type(torch.FloatTensor).to(self.device))

        return loss


class CoherenceLoss(nn.Module):
    def __init__(self, device, weight=None, size_average=True):
        super(CoherenceLoss, self).__init__()
        self.device = device

    def forward(self, inputs, targets, smooth=1):
        # Put tensor on the same device
        inputs.to(self.device)
        targets.to(self.device)

        # comment out if your model contains a sigmoid or equivalent activation layer
        # if model_params.BASELINE:
        #inputs = torch.sigmoid(inputs)
        #inputs = torch.tanh(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice