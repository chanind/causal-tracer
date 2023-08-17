from transformers import GPT2TokenizerFast, GPT2LMHeadModel

from causal_tracer.causal_tracing.CausalTracer import CausalTracer
from causal_tracer.causal_tracing.plot_hidden_flow_heatmap import (
    plot_hidden_flow_heatmap,
)


# It's hard to give a rigorous test for this since it's drawing an image,
# but we can at least make sure it runs without error in the basic case.
def test_plot_hidden_flow_heatmap_doesnt_error(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    tracer = CausalTracer(model, tokenizer)
    hidden_flow = tracer.calculate_hidden_flow("Bill Gates is the CEO of", samples=1)
    plot = plot_hidden_flow_heatmap(hidden_flow, show=False)
    assert plot is not None
