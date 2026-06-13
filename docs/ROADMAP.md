# Chromors Roadmap

Este documento descreve a visão de longo prazo e os próximos passos para o **Chromors**. O objetivo final é integrar o engine como o coração do editor de fotos **Pixors**, garantindo que o núcleo (core) seja robusto, performático e altamente extensível antes de partirmos para features evolutivas e de inteligência artificial.

O roadmap está dividido em duas fases principais: **Estabilização do Core** e **Backlog Evolutivo**.

---

## 🟢 Fase 1: Estabilização & Core "Redondo"
*Objetivo: Deixar a fundação do engine impecável, autossuficiente e documentada para ser consumida de forma estável.*

### 1. Paridade de Operações GPU (Fast Preview Sempre)
Para que a edição seja fluida, todas as operações que a CPU faz precisam de um correspondente na GPU.
* [ ] Mapear e implementar as operações faltantes do `libvips` dentro do backend WGPU/Slang.
* [ ] Garantir que o pipeline de preview em tempo real nunca sofra fallback para a CPU de forma inesperada.
* [ ] Assegurar a paridade matemática (nos limites de tolerância) entre a saída do `libvips` e os JIT shaders do Slang.

### 2. Algoritmo de Cache Tiered (VRAM → RAM → Disk)
O algoritmo de cache do antigo `pixors-engine` provou ser sensacional e à prova de falhas. Precisamos portá-lo para a nova arquitetura do Chromors.
* [ ] Implementar a estrutura baseada em identificadores criptográficos (Hashes do DAG).
* [ ] Estruturar a hierarquia de *eviction* e limites de memória para VRAM, sistema (RAM) e persistência em disco.
* [ ] Integrar perfeitamente com os *requests* assíncronos e o materializador.

### 3. Color Management Nativo (In-House)
Atualmente a comunicação de perfis ICC e transformações recai pesadamente sobre o `libvips`, o que gera atritos com os nossos próprios models de cor.
* [ ] Unificar a interface de tratativas de cor (transfer functions, espaços primários, etc).
* [ ] Integrar o parsing e a aplicação de perfis ICC **diretamente no core**, sem depender de crates externos complexos (escrever do zero ou portar a lógica para os JIT shaders e CPU fallback nativo).
* [ ] Garantir que o pipeline JIT aplique correções de cor de forma fusionada no *WorkingDecode/Encode*.

### 4. Documentação e Especificação
Um engine poderoso não serve se for ilegível.
* [ ] Escrever documentação das funções públicas, com guias de uso básicos e avançados.
* [ ] Documentar o fluxo de compilação JIT dos shaders e o sistema de DAG.
* [ ] Especificar detalhadamente os algoritmos principais (matemática de cores, limites, lógicas de blending).

---

## 🚀 Fase 2: Backlog Evolutivo (Pixors Integration & Expansão)
*Objetivo: Transformar o Chromors em um monstro de manipulação de imagem, abraçando IA, Visão Computacional e alta extensibilidade.*

### 1. Suporte e Cache Aprimorado para Vetores (Vello)
O backend do Vello já se provou excelente, mas pode ser otimizado para cenários complexos.
* [ ] Melhorar os wrappers de renderização de vetores.
* [ ] Implementar suporte a cache de primitivas e cenas pré-renderizadas, evitando rasterização desnecessária em mudanças não destrutivas da árvore de vetores.

### 2. Utilitários de Alto Nível (Helpers do Engine)
Abstrações úteis que vivem "sobre" o engine, facilitando a vida do desenvolvedor da interface de usuário.
* [ ] **`LayerStack`**: Uma estrutura que engloba um array de imagens e monta automaticamente a parte do DAG que utiliza a operação `Composite2` sob os panos, abstraindo as camadas do Photoshop.
* [ ] Implementação de construtores fluentes voltados puramente para UX/UI de manipulação.

### 3. Inteligência Artificial e Modelos (via Burn)
A integração com machine learning rodando na VRAM é o futuro da edição de imagens.
* [ ] Integrar o crate `burn` utilizando o backend WGPU, permitindo compartilhamento de *buffers* "zero-copy".
* [ ] **Semantic Segmentation (YOLO/Segment Anything):** Implementar operações nativas para selecionar e mascarar objetos isolados diretamente no editor.
* [ ] Estruturar o pipeline de entrada e saída dos tensores do ML de volta para os Kinds de `Image` ou `Mask2D` do Chromors.

### 4. Algoritmos Nativos de Computer Vision
Trazer ferramentas avançadas usadas por fotógrafos profissionais diretamente para o escopo matemático do Chromors.
* [ ] **Features Detection & Image Alignment**: Detecção de keypoints (SIFT/ORB) para alinhar fotos tiradas em sequência (pano de fundo de HDR).
* [ ] **Focus Stacking**: Combinar profundidades de campo rasas em uma imagem de foco infinito.
* [ ] **Panorama Blending**: Costura suave de múltiplas imagens em perspectiva.

### 5. Extensibilidade de Shaders Externa
O engine precisa permitir injeção de comportamento sem exigir recompilação da biblioteca core.
* [ ] Permitir que aplicações consumidoras (como o Pixors) passem código `.slang` próprio (custom operations).
* [ ] Criar uma interface para que o compilador do Slang dentro do Chromors possa injetar essas lógicas arbitrárias (ex: nodes complexos de *Color Grading* montados visualmente na UI do usuário) no pipeline fundido.
* [ ] Validar inputs e outputs do shader externo contra o sistema de tipos (Kinds) do Chromors.
