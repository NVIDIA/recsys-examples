import importlib
import os

current_dir = os.path.dirname(__file__)
for filename in os.listdir(current_dir):
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = filename[:-3]
        try:
            module = importlib.import_module(f".{module_name}", package=__name__)
        except ImportError:
            # Inference-only envs may lack training extras (dynamicemb, etc.).
            continue
        globals().update(
            {
                name: getattr(module, name)
                for name in dir(module)
                if not name.startswith("_")
            }
        )
