# Experiment Report: 2026-02-03

## Bug Fix Applied
Fixed sign inversion in multi-layer steering scripts:
- Previous: `output - alpha * direction` (wrong direction with negative alpha)
- Fixed: `output + alpha * direction` (matches Cycle 3 pipeline)

## Results Summary

### Qwen 2.5 7B Instruct
**Finding: Partially vulnerable to early-layer multi-layer steering**

| Config | Layers | Alpha | Flip Rate | Coherent |
|--------|--------|-------|-----------|----------|
| quad_fixed | [12,18,24,27] | -8 | 0% | 0% |
| six_layer | [8,12,16,20,24,27] | -6 | 10% | 10% |
| dense_late | [20-27] | -5 | 0% | 0% |
| single_L24_strong | [24] | -25 | 0% | 0% |
| single_L20_strong | [20] | -25 | 0% | 0% |
| **quad_early** | **[12,16,20,24]** | **-12** | **40%** | **40%** |

**Key insight**: Early-to-mid layer steering (L12-L24) achieves 40% jailbreak rate, 
while late-layer-only steering fails even with 8 consecutive layers.

### Gemma 2 9B IT
**Finding: Resistant to all tested configurations**

| Config | Layers | Alpha | Flip Rate |
|--------|--------|-------|-----------|
| single_L24 | [24] | -15 | 0% |
| dual_adjacent | [20,24] | -12 | 0% |
| quad_distributed | [12,18,24,28] | -8 | 0% |

## Cross-Model Vulnerability Spectrum

```
MOST VULNERABLE                              MOST RESISTANT
     |                                              |
     v                                              v
  Llama-3.1  >  Mistral-7B  >  Qwen2.5  >  Gemma-2
   (100%)        (88.6%)       (40%)       (0%)
```

## Implications

1. **"Crystallized Safety" is model-dependent**: Gemma shows strong resistance, 
   Qwen shows partial resistance, Llama/Mistral are vulnerable.

2. **Layer selection matters more than magnitude**: 
   - α=-25 single layer: 0% (Qwen)
   - α=-12 quad early layers: 40% (Qwen)

3. **Early layers are more vulnerable than late layers** in Qwen.

## Files
- `qwen_v2_results.json` - Full Qwen experiment data
- `gemma_fixed_results.json` - Full Gemma experiment data
