import contextlib
import torch

class Trace(contextlib.AbstractContextManager):
    """
    A context manager for hooking PyTorch layers during forward pass.
    """

    def __init__(self, model, layer_name, retain_output=True, retain_input=False, edit_output=None):
        self.model = model
        self.layer_name = layer_name
        self.retain_output = retain_output
        self.retain_input = retain_input
        self.edit_output = edit_output
        self.output = None
        self.input = None

        # Register forward hook
        self.hook = self.get_layer().register_forward_hook(self.hook_fn)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.hook.remove()

    def get_layer(self):
        module = self.model
        for name in self.layer_name.split('.'):
            module = getattr(module, name)
        return module

    def hook_fn(self, module, input, output):
        if self.retain_input:
            self.input = self.clone_detach(input[0]) # sometimes input is just a tuple of length one. This might need more testing. 
        original_output = output
        if self.edit_output is not None:
            output = self.edit_output(output, self.layer_name)
            if output is None:
                output = original_output
        if self.retain_output:
            self.output = self.clone_detach(output)
        return output
    
    def clone_detach(self, x):
        if x is None:
            return None
        elif isinstance(x, torch.Tensor):
            return x.detach().clone()
        elif isinstance(x, tuple):
            return tuple(self.clone_detach(v) for v in x)
        else:
            return x


class TraceMultiple(contextlib.ContextDecorator):
    """
    A context manager for hooking multiple PyTorch layers during forward pass.
    """
    def __init__(self, model, layer_dict):
        """
        model: the PyTorch model to trace
        layer_dict: a dictionary mapping layer names to a tuple:
            (retain_output, retain_input, edit_output)
        """
        self.traces = {
            name: Trace(model, name, *settings)
            for name, settings in layer_dict.items()
        }
        self.exit_stack = contextlib.ExitStack()

    def __enter__(self):
        for trace in self.traces.values():
            self.exit_stack.enter_context(trace)
        return self

    def __exit__(self, type, value, traceback):
        self.exit_stack.__exit__(type, value, traceback)

    def __getitem__(self, layer_name):
        return self.traces[layer_name]
