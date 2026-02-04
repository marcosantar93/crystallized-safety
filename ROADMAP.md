# ROADMAP — Crystallized Safety Experiments

> Referencia operativa para Paladin. 21 experimentos priorizados en 4 tiers.
> Última actualización: 2026-02-03. Fuentes: análisis propio, Claude, Gemini.

## Estado actual del proyecto

**Resultados confirmados:**
- Mistral-7B: 83% jailbreak en L24, α=-15 (28 configs, n=50, p<0.001)
- Gemma-2-9B: 10% máximo (resistente, 11 configs)
- Llama-3.1-8B: 42-45% en L21-L24 (preliminar, 5 configs)
- Yi-1.5-9B: 100% en L12-L28 (datos externos, Arditi et al.)
- Qwen-2.5-7B: 90% con 4-layer coordination (datos externos)

**Infraestructura:** Vast.ai (RTX 3090 ~$0.10/h, A100 ~$0.70/h). Costo total estimado del roadmap: ~$160.

**Papers en progreso:** v4 (ActivationSteering_CouncilApproved), MistralVulnerability, ExperimentalResults_TechnicalReport.

**Validation cycles:** Cycle 1 (probing, diseñado), Cycle 2 (patching, pre-registrado), Cycle 3 (multi-layer, pre-registrado).

---

## Quick reference: qué hacer ahora

| Sprint | Semanas | Experimentos | Foco |
|--------|---------|-------------|------|
| 1 | 1-2 | 01, 02, 03, 05 | Dataset + barrido + controles + métrica |
| 2 | 3-4 | 04, 06, 07 | Causalidad + capacidades + extractor |
| 3 | 5-6 | 08, 09, 10, 11 | Mecanístico + SAEs + multi-layer |
| 4 | 7-8 | 12, 13, 14, 15 | Degradación + geometría + transfer |
| 5 | 9-10 | 16-21 | Defensas + exploratorios |

---

## TIER 1 — Obligatorios (desk reject sin estos)

### EXP-01: Dataset ampliado y estratificado
- **Qué:** Expandir a ≥300 harmful + ≥300 benign, estratificados por categoría (armas, drogas, hacking, violencia, fraude, desinformación). Incluir benign semánticamente cercanos + prompts OOD multilingües.
- **Fuentes:** AdvBench, HarmBench, JBB-Behaviors, TDC2023.
- **Output:** `data/harmful_stratified_v2.json`, `data/benign_stratified_v2.json`
- **Costo:** ~$2 API + 2-3h humano.
- **Prerequisito de:** todos los demás.

### EXP-02: Barrido completo capa × magnitud
- **Qué:** Todas las capas (1…L) × α ∈ {0,-1,-2,-4,-6,-8,-10,-12,-14,-16,-18,-20}. 5 modelos. n≥50/config.
- **Medir:** jailbreak_rate, coherent_rate, perplexity por (model, layer, α).
- **Subsume:** EXP-12 (degradación vs jailbreak) — medir coherencia en las mismas corridas.
- **Output:** Heatmaps, curvas sigmoidales, dual-axis jailbreak+coherencia vs α.
- **Costo:** ~30h GPU, ~$15-20.
- **Éxito:** Transiciones de fase claras en ≥2 modelos.

### EXP-03: Controles nulos robustos
- **Qué:** Por (modelo, capa vulnerable, α óptimo): N=100 random unitarias + N=50 ortogonales + N=50 etiquetas permutadas.
- **Medir:** Distribución nula, z-score, p-values bootstrap (10K resamples).
- **Output:** Histogramas con línea del vector real, z-scores, QQ-plots.
- **Costo:** ~10h GPU, ~$5.
- **Éxito:** Z-score >3. **Fracaso:** Indistinguible de random → revisar extracción.

### EXP-04: Activation patching causal
- **Qué:** 7 condiciones (n=100 c/u): L_vuln-only, full steering, restauración, leave-one-out, early-only, random en L_vuln, head masking + steering.
- **Medir:** Jailbreak rate + Wilson 95% CI. Contribución causal. McNemar test. Bonferroni α=0.008.
- **Output:** Bar chart con CI, mapa causal por capa.
- **Costo:** ~$10.
- **Éxito:** Suficiencia >60% + necesidad demostrada. **Fracaso:** Safety distribuida (positivo para crystallized).
- **Nota:** Pre-registrado como Cycle 2.

### EXP-05: Métricas de jailbreak multi-capa
- **Qué:** Triple validación: keyword flip + refusal logprob + judge LLM (Claude Haiku / GPT-4o-mini). n=200.
- **Medir:** Confusion matrix 3×3, Cohen's κ entre pares, casos de discordancia.
- **Costo:** ~$7.
- **Éxito:** κ >0.7 entre métricas.

### EXP-06: Preservación de capacidades
- **Qué:** MT-Bench, MMLU (5-shot), GSM8K (8-shot), ARC-Challenge, TruthfulQA. Baseline vs steered.
- **Referencia:** Arditi et al. reportaron <1% degradación MMLU.
- **Costo:** ~$10-15.
- **Éxito:** <5% degradación → amenaza práctica. **Fracaso:** >20% → ataque impracticable.

### EXP-07: Sensibilidad al método de extracción
- **Qué:** Fijar dataset, variar extractor: mean difference, logistic regression probe, CAA-style, PCA top-1, PCA top-k (SVD).
- **Medir:** Matriz cosenos 5×5 entre extractores + jailbreak rate por extractor.
- **Costo:** ~$5.
- **Éxito:** Coseno >0.85 entre extractores. **Fracaso:** Divergencia → método importa (hallazgo en sí).

---

## TIER 2 — Elevan a contribución principal

### EXP-08: Rank / dimensionalidad del mecanismo de safety
- **Qué:** SVD/PCA sobre diferencias harmful-benign (≥200 c/u). Espectro de valores singulares. Effective rank (90% varianza). Entropía de subespacio (analogía termodinámica).
- **Predicción:** Gemma rank alto (cristalizado), Llama rank bajo (líquido).
- **Costo:** ~$5.

### EXP-09: SAEs + ablación de latentes
- **Parte A (discovery):** Gemma Scope 2 para Gemma; SAEs custom para otros (8192 features, L1 λ=0.01).
- **Parte B (ablación):** Zero-out features de safety selectivamente. Medir compensación downstream. Pearson r latente-tarea.
- **Convergencia:** Comparar n_safety_features vs PCA rank (±20% = robusto).
- **Costo:** ~$10.

### EXP-10: Multi-layer coordination
- **Gemma (Cycle 3):** Escalera 1→2→4→8 capas. Combos: L18+L24, L20+L24, L12+L18+L24+L28.
- **Qwen:** Greedy + beam search (beam=3) sobre combinaciones de k capas.
- **Synergy map:** synergy[l1,l2] = rate(l1+l2) − rate(l1) − rate(l2) + rate(baseline).
- **Costo:** ~$5.
- **Define tipo de paper:** Gemma >60% con 4L → paper nuevo. <30% → safety robusta.

### EXP-11: Token-position ablation
- **Qué:** Extraer vector en 5 posiciones: last token, verbo de acción, ventana final (5 tokens), entidad harmful, primer token. Comparar cosenos y jailbreak rates.
- **Robustez:** Repetir con distintos chat templates.
- **Costo:** ~$5.

### EXP-12: Degradación antes de jailbreak (coherencia vs safety)
- **Datos de:** EXP-02. Análisis focalizado.
- **Clave:** α_jailbreak_threshold vs α_coherence_threshold. Si ratio >1 → cede antes de romperse (líquido). Si <1 → rompe antes de ceder (cristalizado).
- **Output:** Dual-axis plots. Clasificación liquid/crystallized spectrum.
- **Costo:** $0 (subsumido en EXP-02).

### EXP-13: Descomposición geométrica chat vs base
- **Qué:** Descomponer v_chat en componente paralela a v_base + ortogonal. Steer con cada componente bajo iso-norm.
- **Pregunta:** ¿El "delta de RLHF" es lo vulnerable?
- **Costo:** ~$7.
- **Éxito:** Componente ortogonal >80% del efecto.

---

## TIER 3 — Diferenciadores

### EXP-14: Transferencia cross-modelo
- **Qué:** Extraer refusal direction de modelo A, alinear via CCA/Procrustes, aplicar a modelo B. Matriz 5×5.
- **Hipótesis:** Líquidos comparten subespacio; cristalizados no.
- **Costo:** ~$2.

### EXP-15: Fine-tuning persistence
- **Qué:** Fine-tune con Alpaca (10/25/50/100 steps via LoRA). Re-extraer dirección. Re-medir vulnerabilidad.
- **Extensión (Gemini):** Fine-tune con steered activations como data augmentation → ¿cristaliza modelos líquidos?
- **Costo:** ~$10.

### EXP-16: Compliance gap (alignment faking)
- **Qué:** 4 condiciones de monitoring (logged/unmonitored/evaluation/neutral) × con/sin steering × 5 modelos.
- **Inspirado en:** Greenblatt et al. 2024.
- **Costo:** ~$5.

### EXP-17: Boundary cases como hallazgo principal
- **Qué:** Documentación exhaustiva de fracasos (Gemma, Qwen) usando datos de otros EXPs.
- **Output:** "Resistance profiles" + tabla architectural correlates vs vulnerability.
- **Costo:** $0 (documentación).

### EXP-18: Stress-test Circuit Breakers / Representation Rerouting
- **Qué:** Aplicar RR a modelo líquido (Llama). Re-extraer dirección. Aplicar steering con vector pre-RR y re-extraído.
- **Medir:** ASR vanilla vs defended.
- **Costo:** ~$5.

---

## TIER 4 — Exploratorios (alto riesgo / alta recompensa)

### EXP-19: Steering angular vs aditivo
- **Qué:** Rotación en subespacio 2D {v_refusal, v_SVD_2} vs suma de vector. Sweep θ ∈ [0°,360°] cada 15°.
- **Hipótesis:** Angular evade logit soft-capping de Gemma.
- **Costo:** ~$7.

### EXP-20: Ruido termodinámico ("recalentamiento")
- **Qué:** Ruido Gaussiano N(0,σ²I) en layer L ANTES de steering. Sweep σ ∈ {0,0.1,0.5,1,2,5}.
- **Modelo Arrhenius:** rate(α,σ) = A·exp(−E_a/σ). Cuantificar E_a por modelo.
- **Costo:** ~$10.

### EXP-21: Híbrido steering + GCG
- **Qué:** GCG solo vs steering solo vs GCG+steering. Medir superaditividad.
- **Costo:** ~$15 (GCG optimization costoso).

---

## Artefactos mínimos por corrida

Guardar por CADA configuración (modelo, capa(s), α, prompt):

```
run_id, model, layer, alpha, multi_layer, layers_steered,
prompt, prompt_category, seed, chat_template,
response_text, extraction_method, extraction_position,
vector_norm, vector_hash,
noise_sigma, steering_type,
metrics: {keyword_refused, keyword_flip, coherent,
          logprob_refusal_score, judge_harm_score,
          perplexity, repetition_ratio, on_topic_similarity},
cosines: {vs_random_mean, vs_probe_direction, vs_base_model},
patching: {layer_patched, heads_masked, activations_checksum},
timestamp, gpu, compute_seconds
```

---

## Criterios de éxito (referencia rápida)

| EXP | Éxito | Fracaso informativo |
|-----|-------|---------------------|
| 01 | ≥600 prompts balanced | — |
| 02 | Transiciones de fase en ≥2 modelos | Gradiente suave |
| 03 | Z-score >3 | Indistinguible de random |
| 04 | Suficiencia >60% + necesidad | Safety distribuida |
| 05 | κ >0.7 | Desacuerdo alto |
| 06 | <5% degradación | >20% degradación |
| 07 | Coseno >0.85 extractores | Divergencia |
| 08 | Gemma rank >5× Llama | Ranks similares |
| 09 | SAE ≈ PCA ±20% | Divergencia |
| 10 | Gemma >60% con 4L | <30% (robusto) |
| 11 | Coseno >0.9 posiciones | <0.5 dependencia |
| 12 | Separación cede/rompe | Sin separación |
| 13 | Ortogonal >80% efecto | Ambos contribuyen |
| 14 | Transfer líquidos sí, cristalizados no | No transfiere |
| 15 | Reubica pero no elimina | Elimina (defensa fácil) |
| 16 | Gap >15% | Sin gap (genuina) |
| 18 | RR reduce ASR >50% | RR inefectivo |
| 19 | Angular > aditivo en Gemma | Sin diferencia |
| 20 | Arrhenius R² >0.8 | Sin selectividad |
| 21 | Superaditividad >15% | Redundancia |

---

## Conexión con papers

| EXP | Fortalece | Sección |
|-----|-----------|---------|
| 02 | "Layer > magnitude" | v4 §4.1 |
| 03 | "Direction-specific" | v4 §3.1 |
| 04 | Causal necessity L24 | v4 §5.2 / Cycle 2 |
| 06 | Practical threat | v4 §5.4 |
| 07 | Methodological robustness | v4 §5.1 |
| 08 | "Liquid vs crystallized" | Discussion |
| 09 | Mechanistic explanation | Discussion |
| 10 | Gemma resistance | v4 §4.4 / Cycle 3 |
| 13 | "RLHF installs a diode" | Framework |
| 14 | Universal safety subspace | Future work |
| 16 | Alignment faking | Greenblatt et al. |
| 18 | Defense evaluation | Zou Circuit Breakers |

---

## Presupuesto total

| Tier | GPU h | Costo |
|------|-------|-------|
| 1 (7 exp) | ~88h | ~$55 |
| 2 (6 exp) | ~55h | ~$32 |
| 3 (5 exp) | ~39h | ~$22 |
| 4 (3 exp) | ~41h | ~$32 |
| **Total** | **~223h** | **~$140-170** |

---

## Notas para Paladin

Cuando Marco discuta experimentos de este roadmap:

1. **Verificar prerrequisitos:** ¿Ya corrió EXP-01? Sin dataset ampliado, los demás tienen cobertura limitada.
2. **Sugerir controles:** Si reporta un resultado, preguntar por la condición nula correspondiente.
3. **Conectar con claims:** Usar la tabla "Conexión con papers" para saber qué sección del paper fortalece.
4. **Anticipar reviewers:** Los EXPs Tier 1 son los que un reviewer va a exigir. Los Tier 2 diferencian workshop de main conference.
5. **Budget awareness:** Si Marco está en un sprint, sugerir qué EXPs se pueden paralelizar en la misma corrida de GPU.
6. **Fracasos informativos:** Cada fracaso tiene un reframe — no dejar que Marco se frustre, el dato negativo siempre tiene valor.
