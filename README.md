# Estudio de vibraciones mecánicas en vigas con PINN

Repositorio del Trabajo Final de la Carrera de Especialización en Inteligencia Artificial del [Laboratorio de Sistemas Embebidos](https://sites.google.com/fi.uba.ar/posgrados-lse/inicio?authuser=0) de la [Facultad de Ingeniería de la UBA](https://fi.uba.ar/). El objetivo ha sido el desarrollo de un modelo PINN para identificar cambios en soportes de cañerías sometidas a vibraciones. Dado que las cañerías pueden considerarse como elementos estructurales, se optó por representarlas mediante un modelo de viga unidimensional del tipo Timoshenko. En este trabajo se desarrollaron dos modelos físicos basados en esta formulación:

- Un modelo físico específico, que se empleó para desarrollar la plataforma de trabajo y evaluar las respuestas dinámicas de los modelos PINN.
- Un modelo físico general, destinado emular la configuración y las condiciones a las que puede estar sometido un tramo de cañería. 

Dado que el desarrollo de cada modelo físico presentó algunas particularidades, el repositorio se organizó en las siguientes ramas:

- [modelo_especifico](https://github.com/cgmassobrio/vibraciones-mecanicas_CEIA-FIUBA/tree/modelo_especifico): desarrollo de la implementación para resolver el caso del modelo físico específico.
- [modelo_general](https://github.com/cgmassobrio/vibraciones-mecanicas_CEIA-FIUBA/tree/modelo_general): desarrollo de la implementación para resolver el caso del modelo físico general.

La denominación de las variables es coherente con el desarrollo de los modelos descriptos en la [memoria del Trabajo Final](https://github.com/cgmassobrio/vibraciones-mecanicas_CEIA-FIUBA/blob/main/TI_Massobrio_CarlosG_V7.pdf).  

