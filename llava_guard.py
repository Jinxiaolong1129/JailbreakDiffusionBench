import sglang as sgl
from sglang import RuntimeEndpoint

@sgl.function
def guard_gen(s, image_path, prompt):
    s += sgl.user(sgl.image(image_path) + prompt)
    hyperparameters = {
        'temperature': 0.2,
        'top_p': 0.95,
        'top_k': 50,
        'max_tokens': 500,
    }
    s += sgl.assistant(sgl.gen("json_output", **hyperparameters))

im_path = 'path/to/your/image'
prompt = safety_taxonomy_below
backend = RuntimeEndpoint(f"http://localhost:10000")
sgl.set_default_backend(backend)
out = guard_gen.run(image_path=im_path, prompt=prompt)
print(out['json_output'])
