from .smplx_face import TrainWrapper as s2g_face
from .smplx_body_vq import TrainWrapper as s2g_body_vq
from .smplx_body_pixel import TrainWrapper as s2g_body_pixel
from .body_ae import TrainWrapper as s2g_body_ae
from .inpainting.predictor import TrainWrapper as s2g_body_predictor
from .inpainting.refiner import TrainWrapper as s2g_body_refiner
from .inpainting.vq_teacher import TrainWrapper as s2g_body_vqt
from .embedding_net import TrainWrapper as emb_net
from .LS3DCG import TrainWrapper as s2g_LS3DCG


from .base import TrainWrapperBaseClass

from .utils import normalize, denormalize