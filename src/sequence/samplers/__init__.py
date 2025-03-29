from .curriculum import ZONES_CURRICULUM, LETTER_CURRICULUM, FLATWORLD_CURRICULUM, FLATWORLD_BIG_CURRICULUM
from .curriculum_sampler import CurriculumSampler

curricula = {
    'PointLtl2-v0': ZONES_CURRICULUM,
    'LetterEnv-v0': LETTER_CURRICULUM,
    'FlatWorld-v0': FLATWORLD_CURRICULUM,
    'FlatWorld-big-v0': FLATWORLD_BIG_CURRICULUM,
}
