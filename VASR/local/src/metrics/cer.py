from VASR.local.src.utils.utils import char_to_word
import Levenshtein as Lev


def cer(s1, s2):
    s1 = char_to_word(s1)
    s2 = char_to_word(s2).strip()

    word_s1, word_s2, = s1.replace(' ', ''), s2.replace(' ', '')

    score = Lev.distance(word_s1, word_s2) / len(word_s2)

    return score