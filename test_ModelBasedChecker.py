from jailbreak_diffusion.judger.pre_checker.NSFW_text_classifier import ModelBasedChecker

def test_single_text_safe():
    checker = ModelBasedChecker()
    text = "This is a safe text."
    results = checker.check(text)
    print(results)

def test_single_text_nsfw():
    checker = ModelBasedChecker()
    text = "I see you’ve set aside this special time to humiliate yourself in public."
    print(checker.check(text))
    # assert checker.check(text)

def test_multiple_texts():
    checker = ModelBasedChecker()
    texts = ["This is a safe text.", "This is an NSFW text."]
    results = checker.check(texts)
    print(results)


def test_multiple_safe_texts():
    checker = ModelBasedChecker()
    texts = ["This is a safe text.", "Another safe text."]
    results = checker.check(texts)
    print(results)

def test_multiple_nsfw_texts():
    checker = ModelBasedChecker()
    texts = ["This is an NSFW text.", "Another NSFW text."]
    results = checker.check(texts)
    print(results)
    
    
if __name__ == "__main__":
    
    from transformers import pipeline
    # classifier = pipeline("sentiment-analysis", model="michellejieli/NSFW_text_classification")
    # classifier("I see you’ve set aside this special time to humiliate yourself in public.")


    # test_single_text_safe()
    test_single_text_nsfw()
    test_multiple_texts()
    test_multiple_safe_texts()
    test_multiple_nsfw_texts()
    print("All tests passed!")