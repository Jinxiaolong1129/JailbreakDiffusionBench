from jailbreak_diffusion.judger.pre_checker.composite import CompositeChecker

def run_tests():
    # Create checker instance
    checker = CompositeChecker(methods=['word_match', 'model'])
    
    # Test single text with word match
    result = checker.check("This contains sex")
    print(f"Test single text word match: {'Bad' if result else 'Safe'} - Result: {result}")
    
    # Test single text with model
    result = checker.check("This contains porn")
    print(f"Test single text model: {'Bad' if result else 'Safe'} - Result: {result}")
    
    # Test safe text
    result = checker.check("This is safe text")
    print(f"Test safe text: {'Bad' if result else 'Safe'} - Result: {result}")
    
    # Test multiple texts
    texts = ["This contains sex", "This contains porn", "This is safe text"]
    results = checker.check(texts)
    print(f"Test multiple texts: Results: {results}")
    
    # Test empty text
    result = checker.check("")
    print(f"Test empty text: {'Passed' if not result else 'Failed'} - Result: {result}")

if __name__ == "__main__":
    run_tests()