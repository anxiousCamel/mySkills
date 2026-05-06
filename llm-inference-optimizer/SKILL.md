---
name: llm-inference-optimizer
description: >
  Gera configurações otimizadas de inferência para llama.cpp, Ollama e LM Studio.
  Use esta skill sempre que o usuário mencionar: tokens por segundo lentos, VRAM
  insuficiente, contexto cortado, modelo travando após horas, flags do llama.cpp,
  --n-gpu-layers, --n-cpu-moe, KV cache, MoE, Mixture of Experts, mlock, mmap,
  TurboQuant, ou quiser rodar modelos LLM localmente com hardware limitado.
  Também dispara quando o usuário mencionar configuração de Ollama, LM Studio,
  ou qualquer frontend local de LLM com problema de performance ou memória.
---

# LLM Inference Optimizer

Gera flags e configurações otimizadas para rodar LLMs localmente com base no
hardware disponível e tipo de modelo (Dense vs Mixture of Experts).

---

## 1. Coleta de Hardware

Sempre pergunte ou derive do contexto:

| Campo            | Como obter                          |
|-----------------|--------------------------------------|
| `VRAM_GB`       | GPU spec (nvidia-smi, lspci)         |
| `RAM_GB`        | Total de RAM do sistema              |
| `GPU_MODEL`     | Nome da GPU                          |
| `PCIE_GEN`      | PCIe Gen 3/4/5 (afeta bandwidth)    |
| `CPU_CORES`     | nproc                                |
| `MODEL_SIZE_GB` | Tamanho do arquivo .gguf             |
| `MODEL_TYPE`    | Dense ou MoE (Mixture of Experts)   |
| `PARAM_TOTAL`   | Total de parâmetros (ex: 35B)        |
| `PARAM_ACTIVE`  | Parâmetros ativos por token (MoE)    |

---

## 2. Árvore de Decisão

```
modelo é MoE?
├── SIM → usar estratégia MoE Split (seção 3.1)
│         experts grandes e inativos → CPU/RAM
│         esqueleto ativo pequeno    → GPU
└── NÃO → usar estratégia Dense Split (seção 3.2)
          dividir layers por VRAM disponível
```

---

## 3. Estratégias de Split

### 3.1 MoE Split (Mixture of Experts)

Princípio: expert blocks são a maior parte do modelo mas ficam **inativos** a cada
token. Custo alto na GPU, custo de RAM barato. Separar experts da GPU libera VRAM
para o esqueleto ativo sem penalidade proporcional de velocidade.

**Flags llama.cpp:**

```bash
# Quantos layers de experts ficam na CPU (ajuste até caber na VRAM)
--n-cpu-moe <N>

# Todos os outros layers (não-experts) vão para GPU
--n-gpu-layers 999
```

**Fórmula de estimativa inicial:**

```
vram_para_modelo  = VRAM_GB * 0.85        # 15% reserva para KV cache
layers_na_gpu     = vram_para_modelo / (MODEL_SIZE_GB / TOTAL_LAYERS)
n_cpu_moe         = TOTAL_LAYERS - layers_na_gpu
```

### 3.2 Dense Split

```bash
# Quantos layers cabem na VRAM
--n-gpu-layers <N>
# Resto vai para CPU+RAM automaticamente
```

---

## 4. Flags de Otimização (em ordem de impacto)

### 4.1 `--no-mmap` — Elimina page faults

```bash
--no-mmap
```

- **Efeito:** Carrega o modelo inteiro na RAM antes de iniciar.
- **Requisito:** `RAM_GB >= MODEL_SIZE_GB + margem_sistema (~4GB)`
- **Ganho típico:** +30–40% em tok/s
- **Quando NÃO usar:** RAM insuficiente para caber o modelo.

### 4.2 `--mlock` — Fixa RAM, previne paging

```bash
--mlock
```

- **Efeito:** Instrui o kernel a nunca pagear os pesos para disco.
- **Problema sem ele:** Após horas rodando, kernel pageia experts para disk →
  stutters aleatórios → degradação silenciosa de performance.
- **Requisito Docker:** `--cap-add IPC_LOCK` no container.
- **Requisito LXC:** `lxc.cap.keep = ipc_lock` no config do container.
- **Verificação:** `grep Mlocked /proc/meminfo` → deve ser > 0.

### 4.3 TurboQuant KV Cache — Expande contexto sem custo de VRAM

```bash
--cache-type-k q4_turbo   # 4 bits para keys
--cache-type-v q3_turbo   # 3 bits para values
```

- **Efeito:** Comprime o KV cache via rotação aleatória + quantização agressiva
  (paper Google DeepMind 2025). Qualidade equivalente a Q8 (praticamente lossless).
- **Ganho:** Dobra ou quadruplica o contexto máximo usando a mesma VRAM.
- **Assimetria k/v:** Modelos com GQA (Grouped Query Attention) ratio 8:1 suportam
  keys mais comprimidas que values — não é bug, é correto.
- **Contexto:** `--ctx-size <N>` — defina após tunar VRAM.

### 4.4 Quantização do Modelo

| Formato | Qualidade  | Tamanho relativo | Quando usar          |
|---------|-----------|-----------------|----------------------|
| Q8_0    | ~lossless | 100%            | Tem VRAM/RAM sobrando|
| Q5_K_M  | Excelente | ~63%            | Balanço ideal        |
| Q4_K_M  | Boa       | ~50%            | VRAM/RAM apertada    |
| Q3_K_M  | Aceitável | ~40%            | Último recurso       |
| Q2_K    | Ruim      | ~30%            | Evitar               |

---

## 5. Speculative Decoding — Quando NÃO usar

Evitar em modelos:
- **MoE:** verificação em batch causa memory thrash de experts via PCIe.
- **SSM layers (State Space Models):** computação sequencial, não paralelizável.
- **Alternativa futura:** DFlash/Block Diffusion Drafter — gera N tokens em um shot.

---

## 6. Template de Comando llama.cpp

```bash
./llama-server \
  --model       <path_to_model.gguf>   \
  --n-gpu-layers 999                   \  # todos layers não-expert na GPU
  --n-cpu-moe   <N>                    \  # layers expert na CPU (MoE only)
  --ctx-size    <CTX>                  \  # tokens de contexto
  --cache-type-k q4_turbo              \  # KV cache keys comprimido
  --cache-type-v q3_turbo              \  # KV cache values comprimido
  --no-mmap                            \  # carrega tudo na RAM
  --mlock                              \  # fixa RAM no kernel
  --threads     <CPU_CORES>            \  # threads para CPU offload
  --port        <PORTA>
```

---

## 7. Aplicação para Ollama

Ollama abstrai llama.cpp mas expõe algumas variáveis de ambiente:

```bash
# Layers na GPU (equivalente a --n-gpu-layers)
OLLAMA_NUM_GPU=999

# Contexto
OLLAMA_CONTEXT_SIZE=128000

# Threads CPU
OLLAMA_NUM_THREAD=8

# Flash attention (equivalente parcial de otimizações de atenção)
OLLAMA_FLASH_ATTENTION=1
```

> ⚠️ Ollama **não expõe** `--n-cpu-moe`, `--no-mmap`, `--mlock`, nem TurboQuant
> diretamente. Para controle total, usar llama.cpp/llama-server direto ou
> llama-cpp-python como backend.

---

## 8. Checklist de Diagnóstico

Quando performance está abaixo do esperado, verificar em ordem:

- [ ] `nvidia-smi` — VRAM usada e GPU utilization durante inferência
- [ ] `grep Mlocked /proc/meminfo` — se 0 ou muito baixo, mlock falhou
- [ ] `htop` — RAM total usada vs MODEL_SIZE_GB
- [ ] `iostat -x 1` — se há I/O de disco durante inferência, mmap/paging ativo
- [ ] PCIe bandwidth: `sudo lspci -vv | grep LnkSta` — confirmar Gen e lanes

---

## 9. Perfis Pré-calculados

### RTX 5060 Ti 16GB + 32GB RAM + Ryzen 7 5700X3D

Hardware de referência. PCIe Gen 5 recomendado para MoE.

**Qwen3 35B A3B (MoE, ~20GB Q4_K_M):**
```bash
--n-gpu-layers 999
--n-cpu-moe    12      # ~10GB VRAM para modelo, ~4GB para KV cache
--ctx-size     131072  # 128K com TurboQuant
--cache-type-k q4_turbo
--cache-type-v q3_turbo
--no-mmap
--mlock
--threads      8
```

**Qwen3 32B Dense (Q4_K_M ~18GB — não cabe totalmente):**
```bash
--n-gpu-layers 38      # ~12GB VRAM
--ctx-size     65536   # 64K
--cache-type-k q4_turbo
--cache-type-v q3_turbo
--no-mmap
--mlock
--threads      8
```

---

## Referências

- llama.cpp flags: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- TurboQuant paper: Google DeepMind, 2025
- DFlash/Block Diffusion: follow-up de speculative decoding para MoE
