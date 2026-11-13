# Guía de ejecución y aprendizaje por refuerzo

Esta guía explica el proyecto desde que descargas el ZIP hasta que ejecutas Paw Sweep con el agente entrenado. La simulación es propia e inspirada en el combate de Toriel; no abre, modifica ni automatiza Undertale.

## 1. ¿Qué simula el proyecto?

El entorno es el Bullet Board: un tablero donde se mueve el SOUL y aparecen patrones de ataque. Cada episodio usa una semilla para que la selección aleatoria de ataques y sus variantes se pueda reproducir.

Los ataques disponibles son:

- **Fire Helix:** corrientes de fuego gruesas y un corredor seguro.
- **Fire Waves:** corrientes entrecruzadas, huecos y, algunas veces, una variante difícil con tablero más corto y zonas seguras pequeñas.
- **Paw Sweep:** una pata barre el tablero y deja bolas de fuego que luego persiguen al SOUL; en rondas posteriores puede aparecer una segunda pata.
- **Faltering:** bolas irregulares que se alejan del SOUL y no causan daño.

Los movimientos y colisiones son aproximaciones para entrenar y experimentar, no una réplica exacta del juego.

El SOUL comienza con 20 HP. Si un ataque normal lo deja en 2 HP o menos, el ataque se interrumpe y Toriel pasa a Faltering. En un combate completo, cada ronda terminada aumenta el contador de Spares; al llegar a 11, los ataques siguientes son Faltering, y al llegar a 12 Toriel deja de atacar.

## 2. Elementos de aprendizaje por refuerzo

### Entorno

El entorno (`TorielEnv`) mantiene el tablero, HP, ataque actual, proyectiles y contador de Spares. En cada paso recibe una acción, mueve el SOUL, avanza el ataque y calcula recompensa y daño. La simulación usa pasos fijos; el renderizado de Pygame es opcional.

### Agente

El agente es el SOUL: el corazón rojo controlado por el programa. El algoritmo Q-Learning aprende qué acción conviene elegir para cada estado. Toriel no es otro agente que aprenda: sus ataques y reglas los decide el entorno.

### Estados observados

El entorno entrega 12 valores numéricos normalizados entre 0 y 1:

| Índice | Información |
| --- | --- |
| 1 | Posición horizontal del SOUL respecto al ancho del tablero |
| 2 | Posición vertical del SOUL respecto a la altura |
| 3 | HP actual respecto al HP máximo |
| 4 | Ataque: Fire Helix, Fire Waves, Paw Sweep o Faltering |
| 5 | Fase/progreso del ataque actual |
| 6 | Spares acumulados respecto al máximo de 12 |
| 7 | Posición relativa al corredor de Fire Helix; en otros ataques usa un valor neutro |
| 8 | Ancho del corredor de Fire Helix; en otros ataques usa un valor neutro |
| 9-12 | Distancia resumida al peligro más cercano arriba, abajo, izquierda y derecha; un valor menor indica más proximidad |

La tabla Q no guarda cada proyectil por separado. Discretiza la posición, HP, tipo/fase del ataque y un resumen de las direcciones peligrosas para mantener pequeño el espacio de aprendizaje. Aunque el contador de Spares aparece en la observación, la discretización actual no lo usa como dimensión de la tabla.

### Acciones

El agente dispone de cinco acciones:

| Acción | Movimiento |
| --- | --- |
| 0 | Quedarse quieto |
| 1 | Arriba |
| 2 | Abajo |
| 3 | Izquierda |
| 4 | Derecha |

No hay diagonales en esta versión.

### Políticas utilizadas

Una política es la regla que decide qué acción elegir en cada estado. Este proyecto usa varias, según la etapa:

- **Entrenamiento Q-Learning, ε-greedy:** con probabilidad ε el agente explora eligiendo una acción aleatoria; de lo contrario, elige la acción con mayor valor Q conocido para ese estado. ε empieza en `1.0`, disminuye un factor `0.9995` al final de cada episodio y no baja de `0.03`. Al comenzar los episodios de combate completo vuelve como mínimo a `0.30` para permitir más exploración.
- **Demo con modelo, codiciosa (greedy):** usa la acción de mayor valor Q, sin exploración aleatoria. Por eso repetir el mismo escenario y semilla suele producir las mismas decisiones.
- **Baseline de evaluación, aleatoria:** elige uniformemente entre las cinco acciones. Sirve como referencia para comprobar si Q-Learning supera decisiones sin estrategia.
- **Selección de ataques en combate completo:** el entorno elige al azar entre Fire Helix, Fire Waves y Paw Sweep al iniciar cada ronda. La semilla permite reproducir esa secuencia. En la evaluación de un patrón aislado, el ataque se fija mediante `--attack`.

### Recompensas y penalizaciones

| Situación | Recompensa |
| --- | ---: |
| Paso sin colisión | +0.05 |
| Recibir un impacto | -10 |
| Morir | -20 adicionales |
| Completar un patrón aislado sin recibir daño | +5 adicionales |
| Completar una ronda sin daño en un combate completo | +0.5 adicionales |
| Llegar al duodécimo Spare y terminar el combate | +5 adicionales |

Las bolas de fuego de los ataques normales hacen 3 HP en Paw Sweep y 4 HP en Fire Helix/Fire Waves. Tocar una pata hace 4 HP. Faltering no hace daño. Estos valores de daño no son las recompensas de Q-Learning: el entorno los usa para reducir HP; la tabla anterior describe la señal que recibe el agente.

## 3. Descargar y preparar el ZIP en Windows

1. Descarga el ZIP y usa **Extraer todo**. No ejecutes Python desde dentro del ZIP.
2. Abre PowerShell en la carpeta extraída. Debes estar en la raíz del proyecto: allí existen `pyproject.toml`, `README.md`, `src` y `tests`.
3. Crea un entorno virtual local para mantener las dependencias del proyecto separadas:

```powershell
py -m venv .venv
```

4. Instala el proyecto y sus dependencias usando directamente Python del entorno virtual:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[render,test]"
```

No es obligatorio activar `.venv`. Si PowerShell bloquea `Activate.ps1` por la política de ejecución, puedes continuar con el comando anterior y anteponer `\.venv\Scripts\python.exe` a cada comando Python, como en los ejemplos siguientes. No hace falta cambiar la política del sistema.

## 4. Entrenar Q-Learning

```powershell
.\.venv\Scripts\python.exe -m toriel_rl.training --episodes 2000 --full-combat-episodes 500 --seed 7
```

Qué significa:

- `--episodes 2000`: cantidad de episodios de práctica aislada. El entrenamiento rota Fire Helix, Fire Waves y Paw Sweep.
- `--full-combat-episodes 500`: después hace 500 episodios de combate completo para ajustar el agente a las rondas encadenadas.
- `--seed 7`: fija la semilla inicial para hacer reproducible el entrenamiento.

Al finalizar, se crean:

- `results/q_table.json`: tabla Q aprendida, que luego carga la demo y la evaluación.
- `results/training.csv`: métricas de los episodios de entrenamiento.

La línea `Total episodes: 2500` confirma que se hicieron 2,000 + 500 episodios. `Recent completion rate` y `Recent mean damage` resumen los últimos episodios de entrenamiento; no son la evaluación independiente del modelo.

## 5. Evaluar y entender los resultados

```powershell
.\.venv\Scripts\python.exe -m toriel_rl.evaluation --model results/q_table.json --episodes 100
```

El programa compara Q-Learning con una política aleatoria usando las mismas semillas. Muestra cada ataque por separado y `full_combat`:

- `completion`: porcentaje de patrones aislados que completó o de combates completos que terminó.
- `mean damage`: HP promedio perdido por episodio. Menos es mejor.
- `mean HP`: HP promedio al terminar.
- `reward`: recompensa acumulada promedio. Más alta suele ser mejor, pero conviene leerla junto con daño y completitud.

La evaluación crea `results/evaluation.csv`. En una ejecución de referencia del proyecto se observó:

| Prueba | Aleatorio: completitud / daño medio | Q-Learning: completitud / daño medio |
| --- | --- | --- |
| Fire Helix | 41% / 14.28 HP | 100% / 3.68 HP |
| Fire Waves | 80% / 14.84 HP | 93% / 11.36 HP |
| Paw Sweep | 100% / 9.21 HP | 100% / 1.35 HP |
| Combate completo | 20% / 19.65 HP | 26% / 19.65 HP |

Son resultados de una corrida concreta, no valores garantizados: dependen del modelo entrenado, semillas y configuración. Indican que el agente esquiva mejor patrones aislados; todavía tiene mucho margen para mejorar en un combate completo.

## 6. Ejecutar Paw Sweep con el modelo

```powershell
.\.venv\Scripts\python.exe -m toriel_rl.main --model results/q_table.json --attack paw_sweep --attack-only --seed 7

.\.venv\Scripts\python.exe -m toriel_rl.main --model results/q_table.json --seed 7 (ataques aleatorios)
```

- `--model`: carga la tabla Q creada durante el entrenamiento.
- `--attack paw_sweep`: elige el ataque que se mostrará.
- `--attack-only`: ejecuta solo ese patrón, no las 12 rondas del combate.
- `--seed 7`: hace reproducible esa prueba.

La ventana muestra el SOUL y los peligros. Al terminar, la terminal informa el motivo, HP restante y Spares. Si omites `--attack-only`, el agente juega el combate completo; si omites `--model`, la demo usa acciones aleatorias.

También puedes cambiar `paw_sweep` por `fire_helix`, `fire_waves` o `faltering` para inspeccionar otro patrón.

## 7. Ejecutar las pruebas

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

El resultado esperado al final es `OK`. Si cambias la observación del entorno y el modelo informa que su formato es antiguo, vuelve a entrenar para crear una tabla Q compatible.

## 8. Vitrina y arena en vivo

El repositorio incluye cinco modelos PPO/SAC, cada uno especializado en una
tarea. `slalom_focus` es el modelo recomendado para la demo de slalom; no hay un
modelo que sea el mejor para todas las escenas. Puedes medirlo en su tarea con:

| Modelo | Algoritmo | Tarea |
|---|---|---|
| `slalom_focus` | PPO | Esquivar el slalom usando foco: alternar entre moverse y quedarse quieto según el color. |
| `survival_rain` | PPO | Sobrevivir a una lluvia de proyectiles. |
| `endurance` | SAC | Sobrevivir durante una partida larga mientras aumenta la dificultad. |
| `mixup` | SAC | Adaptarse a patrones y niveles que cambian entre episodios. |
| `collapsing_arena` | SAC | Esquivar mientras el área de juego se encoge. |

```powershell
.\.venv\Scripts\python.exe -m toriel_rl.sandbox_training --eval-only models/slalom_focus/sandbox_ppo.zip --mode focus_precision --scenario slalom_switching --level 3 --eval-episodes 50 --seed 10000
```

```powershell
# vitrina: las 24 escenas en orden (3 ataques de Toriel, 15 patrones, 6 escenarios)
# con tarjeta de título de 2 s y 15 s de acción por escena; política aleatoria
.\.venv\Scripts\python.exe -m toriel_rl.showcase

# la misma vitrina con la tabla Q o el modelo PPO incluido
.\.venv\Scripts\python.exe -m toriel_rl.showcase --model results/q_table.json
.\.venv\Scripts\python.exe -m toriel_rl.showcase --model models/slalom_focus/sandbox_ppo.zip

# arena en vivo: escena aleatoria por episodio, hasta cerrar la ventana
.\.venv\Scripts\python.exe -m toriel_rl.arena --model models/slalom_focus/sandbox_ppo.zip
.\.venv\Scripts\python.exe -m toriel_rl.arena --random
```

Para probar los otros modelos, ejecuta **una sola línea cada vez**, desde la
carpeta raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe -m toriel_rl.arena --model models/survival_rain/sandbox_ppo.zip
.\.venv\Scripts\python.exe -m toriel_rl.arena --model models/endurance/sandbox_sac.zip
.\.venv\Scripts\python.exe -m toriel_rl.arena --model models/mixup/sandbox_sac.zip
.\.venv\Scripts\python.exe -m toriel_rl.arena --model models/collapsing_arena/sandbox_sac.zip
```

Estos comandos ejecutan los modelos, no los entrenan. No es necesario probarlos
todos. Cada modelo aprende una tarea distinta; como la arena elige escenas al
azar, puede usar una política aleatoria de respaldo en escenas que el modelo no
cubre.

- Cada escena empieza con una tarjeta: nombre (p. ej. `A3 Blue Collapsing Ring`)
  y regla de color (p. ej. `Blue: stay still to survive`).
- La barra inferior muestra el nombre de la escena, la recompensa del episodio,
  la barra de HP y el indicador `IFR` de iFrames.
- Un `.json` aplica la tabla Q a los ataques de Toriel y un `.zip` de PPO/SAC al
  sandbox; en las escenas que el modelo no cubre se usa la política aleatoria.
- `--seconds` (15) y `--title-seconds` (2 en vitrina, 1 en arena) ajustan las
  duraciones, `--loop` repite la vitrina y `--episodes N` limita la arena.
