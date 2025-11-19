# Estudio de vibraciones mecánicas en vigas con PINN

Repositorio del trabajo final de CEIA "Desarrollo de modelo para identificar cambios en soportes de cañerías sometidas a vibraciones". Dado que las cañerías pueden considerarse como elementos estructurales, se optó por representarlas mediante un modelo de viga unidimensional del tipo Timoshenko. En este trabajo se desarrollaron dos modelos físicos basados en esta formulación:

- Un modelo físico específico, que se empleó para desarrollar la plataforma de trabajo y evaluar las respuestas dinámicas de los modelos PINN.
- Un modelo físico general, destinado emular la configuración y las condiciones a las que puede estar sometido un tramo de cañería. 

Dado que el desarrollo de cada modelo físico presentó algunas particularidades, el repositorio se organizó en las siguientes ramas:

- modelo_especifico: desarrollo de la implementación para resolver el caso del modelo físico específico.
- modelo_general: desarrollo de la implementación para resolver el caso del modelo físico general.



