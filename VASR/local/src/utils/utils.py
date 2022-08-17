
def convert_to_strings(inverse_map, out):
    results = []
    for i in range(len(out)):
        y = out[i]
        mapped_pred = [inverse_map[j] for j in y]
        results.append(mapped_pred)
    return results


def char_to_word(output_list):
    word_string = "".join(output_list)
    return word_string
