# import 'spaCy' library
import spacy

# similarity list
similarity_cat_monkey_banana = [
    "Cats, monkeys, and bananas may seem like an odd combination of things, but there are actually several interesting similarities between them. Despite their vast differences, these similarities illustrate their close bond with one another and nature.", 
    "Monkeys and cats are both mammals, and they share a common ancestor millions of years ago, while bananas are a type of fruit that evolved from wild plant species that have existed for millions of years. However, various cultures around the world associate cats, monkeys, and bananas with a variety of symbolic meanings.", 
    "Cats are often associated with mystery, independence, and luck, as are monkeys, which are sometimes associated with mischief, intelligence, and curiosity, and bananas with happiness, energy, and romance, among other things. But there are also several funny similarities between cats, monkeys, and bananas, and they all have a sense of humor.", 
    "Cats make us laugh with their silly antics, monkeys are known for their mischievous behavior, and who hasn't giggled when you saw a banana painted with a funny face?", 
    "Interestingly, the cats, monkeys, and bananas are all hanging. Cats love to hang out on windowsills, monkeys love to hang from trees, and bananas like to hang from bunches. All of them make distinct sounds: cats meow, monkeys chatter and screech, and bananas do not make sound unless squished.", 
    "It's not surprising that they can all be slippery too. Cats can be sneaky and elusive, monkeys can swing away in a flash, and bananas can be slippery when they're ripe and ready to eat. The three of them are all masters at climbing too. Cats climb trees and jump to great heights, monkeys climb trees and swing from branch to branch, and bananas grow on trees and are often harvested by skilled climbers.", 
    "There is a distinctive scent for each of them. Cats have a distinctive scent that can be either pleasant or unpleasant, monkeys have a musky smell, and bananas have a sweet, fruity aroma. There are also many ways to use them creatively. Cats can be used as inspiration for art and literature, monkeys can be trained to perform in circuses and movies, and bananas can be used to make a variety of dishes and desserts.", 
    "However, while cats, monkeys, and bananas may seem like an odd combination of things, there are actually several interesting similarities between them. These similarities illustrate their close bond with one another and nature and show that, despite their differences, they are all important and valuable parts of the world around us."
]

# load the 'sm' model
nlp_sm = spacy.load("en_core_web_sm")
# load the 'md' model
nlp_md = spacy.load("en_core_web_md")


# loop through each sentence to calculate similarity
for sentence in similarity_cat_monkey_banana:

    # process the sentence using both models
    doc_sm = nlp_sm(sentence)
    doc_md = nlp_md(sentence)

    # calculate similarity between sentence and "cat monkey banana" using both models
    similarity_sm = doc_sm.similarity(nlp_sm("cat monkey banana"))
    similarity_md = doc_md.similarity(nlp_md("cat monkey banana"))

    # print the similarity results
    print(f"Similarity with 'en_core_web_sm': {similarity_sm}")
    print(f"Similarity with 'en_core_web_md': {similarity_md}")
    print('\n')