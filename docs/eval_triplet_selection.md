# Selección de tripletes para el experimento perceptual (HMA)

> Documento metodológico para la tesis. Describe cómo se seleccionaron los 10
> tripletes (anchor + opción A + opción B) usados en el experimento de
> *Human-Model Agreement* (HMA). Script asociado:
> [`backend/scripts/select_eval_triplets.py`](../backend/scripts/select_eval_triplets.py).

## 1. Qué es un triplete

Cada ítem del experimento perceptual es un **triplete**:

- **anchor**: canción de referencia.
- **opción A** y **opción B**: dos candidatas.
- Tarea del humano: *"¿cuál de las dos suena más parecida al anchor?"*

El humano escucha sin ver etiquetas. Su voto mayoritario por triplete es la
**verdad de referencia** (ground truth) contra la que se compara cada modelo.
Como el ground truth es sobre las **canciones**, es independiente del modelo:
se puede evaluar el HMA de *cualquier* modelo que tenga embeddings de esos
tracks, no solo de los 4 que formaron el oráculo de selección.

## 2. Cómo se seleccionaron (metodología real)

Los tripletes **no** se eligieron por coincidencia de género ni a mano. Se
generaron con un procedimiento basado en distancia y desacuerdo entre modelos.
Se insertaron en `eval_triplets` el 2026-04-17.

### 2.1 Oráculo de distancia: promedio coseno de 4 modelos

La "similitud" usada para rankear candidatos **no** vino de un solo modelo de
referencia, sino del **promedio de la distancia coseno sobre 4 modelos**
simultáneamente:

| Modelo | Dataset |
|--------|---------|
| `musicnn` | `msd` |
| `vgg` | `msd` |
| `whisper_contrastive_multilabel` | `base` |
| `musicnn_multisignal` | `msd` |

```
avg_cos_dist(anchor, other) = mean_m( cosine_distance( emb_m[anchor], emb_m[other] ) )
```

Usar el promedio evita que el sesgo de un solo modelo determine qué tan
"similares" se consideran dos canciones.

### 2.2 Pools por percentil (la perilla de dificultad)

Para cada anchor se ordenan todos los demás tracks por `avg_cos_dist` y se
definen dos bandas:

| Pool | Banda de percentil | Rol |
|------|-------------------|-----|
| `close_pool` | 0 – 15 % (más similar) | candidato **positivo** (la opción "correcta" esperable) |
| `medium_pool` | 15 % – 35 % | **distractor difícil** (medianamente similar, no obvio) |

Una opción se toma de cada pool. La clave para que el triplete **no sea
trivial** es que el distractor venga de la banda 15–35 % y no de muy lejos: así
ambas opciones son plausibles y la elección exige discriminación real.

### 2.3 Filtro de desacuerdo entre modelos

Un triplete se conserva **solo si los 4 modelos no coinciden unánimemente** en
cuál opción está más cerca del anchor:

```
votes = [ 'pos' si pos más cerca que neg, si no 'neg'  (por cada modelo) ]
if len(set(votes)) < 2:   # los 4 votaron igual → poco informativo
    descartar
```

Estos **tripletes de desacuerdo** son los más informativos: son justo los casos
donde el juicio humano puede arbitrar entre modelos que discrepan.

### 2.4 Randomización y reproducibilidad

- La posición A/B se asigna 50/50 al azar para evitar sesgo de posición.
- `random.seed(42)`; se muestrean hasta 80 anchors candidatos hasta juntar 10
  tripletes que pasen el filtro de desacuerdo.
- Solo se consideran nombres de archivo FMA (`^\d{6}\.mp3$`) presentes en los
  4 modelos.

## 3. Aclaración importante para la tesis

La terminología informal **"easy / soso / hard"** que se manejó en
conversación **no se almacenó como etiqueta** en la base de datos ni gobernó la
selección. La dificultad real proviene de:

1. la **banda de percentil** del distractor (close vs medium), y
2. el **filtro de desacuerdo** entre los 4 modelos.

Cualquier categorización posterior por género (p.ej. "los 3 de géneros
distintos") es un **análisis a posteriori** sobre los tripletes ya elegidos, no
el criterio de selección. Debe reportarse así para no inducir a error.

## 4. Reproducir

```bash
cd backend
# Vista previa (no inserta):
python scripts/select_eval_triplets.py --db db/data.db --dry-run

# Generar e insertar 10 tripletes:
python scripts/select_eval_triplets.py --db db/data.db --n-triplets 10 --seed 42
```

Requiere que la tabla `embeddings` ya tenga los vectores de los 4 modelos del
oráculo (correr `flask preprocess-all` antes). El `--seed 42` reproduce la
muestra original siempre que el conjunto de embeddings sea el mismo.

## 5. Evaluar el HMA de todos los modelos

El acuerdo humano-modelo se calcula comparando, para cada triplete, la opción
que elige cada modelo (menor distancia coseno al anchor) contra el voto
mayoritario humano. Es evaluable para **todos** los modelos con embeddings de
esos tracks (no solo los 4 del oráculo). Ver el comando `compute-hma` y el
notebook de análisis (`REPRODUCE.md`, sección C).
