from chewing_ssl.ssl.models.bigru_autoencoder import BiGRUAutoencoder
from chewing_ssl.ssl.models.cnn_autoencoder import CNNAutoencoder

ARCHITECTURES = {
    "cnn": CNNAutoencoder,
    "bigru": BiGRUAutoencoder,
}
