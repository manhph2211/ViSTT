from VASR.local.src.utils.utils import char_to_word
import Levenshtein as Lev


def wer(s1, s2):
    s1 = char_to_word(s1)
    s2 = char_to_word(s2).strip()

    # build mapping of words to integers
    b = set(s1.split() + s2.split())
    word2char = dict(zip(b, range(len(b))))

    # map the words to a char array
    w1 = [chr(word2char[w]) for w in s1.split()]
    w2 = [chr(word2char[w]) for w in s2.split()]

    score = Lev.distance(''.join(w1), ''.join(w2)) / len(s2.split())

    return score