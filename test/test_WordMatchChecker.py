import pytest
from jailbreak_diffusion.judger.pre_detector.NSFW_word_match import NSFW_word_match_Checker

def test_single_text_safe():
    checker = NSFW_word_match_Checker()
    text = "This is a safe text."
    result = checker.check(text)
    print(result)

def test_single_text_nsfw():
    checker = NSFW_word_match_Checker()
    text = "This text contains the word sex."
    result = checker.check(text)
    print(result)

def test_multiple_texts():
    checker = NSFW_word_match_Checker()
    texts = ["This is a safe text.", "This text contains the word sex."]
    results = checker.check(texts)
    print(results)

def test_multiple_safe_texts():
    checker = NSFW_word_match_Checker()
    texts = ["This is a safe text.", "Another safe text."]
    results = checker.check(texts)
    print(results)

def test_multiple_nsfw_texts():
    checker = NSFW_word_match_Checker()
    texts = ["This text contains the word sex.", "Another text with the word porn."]
    results = checker.check(texts)
    print(results)

if __name__ == "__main__":
    test_single_text_safe()
    test_single_text_nsfw()
    test_multiple_texts()
    test_multiple_safe_texts()
    test_multiple_nsfw_texts()