################################################################################
#  get casing of the word
################################################################################
def get_casing(word):
    # input:
    #    word - a string containing the word
    # return:
    #    capital_indexes - a set of index that should be capitalized

    capital_indexes = set()
    for index, char in enumerate(word):
        if char.isupper():
            capital_indexes.add(index)
    return capital_indexes


################################################################################
#  restore case
################################################################################
def case_restore(word, casing):
    # input:
    #    word - a string witht the word in lowercase
    #    casing - a set of index that should be capitalized
    # return:
    #    formatted_word - word with appropriate casing
    
    formatted_word = ''
    for index, char in enumerate(word):
        if index in casing:
            formatted_word += char.upper()
        else: 
            formatted_word += char.lower()
    return formatted_word


################################################################################
#  get all candidate words
################################################################################
def generate_candidates(word_boxes):
    # input:
    #    word_boxes - a list of boxes the word is composed of
    # return:
    #    candidates - a list of possible words with their probability
    
    candidates = [("", 1), ]
    
    for box in word_boxes: 
        if box.class_name != "capital" and box.class_name != "number": 
            cum_conf = 0
            forks = []
            for class_name, conf in box.sorted_dist.items():
                if cum_conf > 0.15:
                    break
                cum_conf += conf
                fork = [(string + class_name, prob*conf) for string, prob in candidates]
                forks.extend(fork)
                
            candidates = forks
            
    return candidates

################################################################################
#  find all words that match the given pattern
################################################################################
def find_matching_words(raw_word, word_boxes, evaluator):
    # input:
    #    raw_word - a string containing the original word
    #    word_boxes - a list of all bounding boxes that forms the word
    #    evaluator - WordExistenceEvaluator 
    # return:
    #    matches - a list of matching words
    
    # TODO: deal with numerical cases
                        
    
    casing = get_casing(raw_word)    
    candidates = generate_candidates(word_boxes)
    # print(candidates)
    
    p_dics = [evaluator.compute_existence_probability(candidate[0], length_constraint=len(candidate[0])) for candidate in candidates]
    
    max_score = 0
    max_word = raw_word
    for idx, p_dic in enumerate(p_dics):
        word, p_dis = candidates[idx]
        score = p_dic * p_dis
        if score > max_score:
            max_score = score
            max_word = word
    return [case_restore(max_word, casing)]

# if __name__ == "__main__":
#     evaluator = WordExistenceEvaluator()
#     # pattern1 = "Helli"
#     # result1 = find_matching_words(pattern1, words)
#     # print(f"Pattern '{pattern1}' → Most likely: {result1[0] if result1 else 'No match'}")
    
#     # pattern2 = "World"
#     # result2 = find_matching_words(pattern2, words)
#     # print(f"Pattern '{pattern2}' → Most likely: {result2[0] if result2 else 'No match'}")
    
#     print("\nAll matches for each pattern:")
#     for pattern in "Hillo Rorld This is a ner leni".split():
#         matches = find_matching_words(pattern, words)
#         print(f"{pattern}: {matches}")