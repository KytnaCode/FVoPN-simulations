# Fair Voting on Permissionless Network Simulations

Este proyecto contiene las simulaciones realizadas para el proyecto Fair Voting on
Permissionless Networks presentado en la [Fería Mexicana de Ciencias e Ingeneria (FEMECI)](https://app.femeci.mx/) Sinaloa 2026 por el alumno Alejandro Paz Gómez.

## Dependencias

Se utilizó [nix](https://nixos.org/) para entornos de desarrollo reproducibles, 
[uv](https://github.com/astral-sh/uv) para gestionar las dependencias del proyecto y 
los entornos virtuales.

Si se está utilizando nix, se necesita activar la opción experimental de los *flakes*,
una vez hecho, se puede entrar en un entorno de desarrollo con todas las dependencias
instaladas:

```sh
nix develop
```

Si no se utiliza nix, las siguientes herramientas deben estar instaladas:

* GNU Make
* uv

## Uso

Distintos invariantes y propiedades son demostradas en forma de test unitarios en 
la carpeta [tests](./tests/). Se prueban estas propiedades del sistema para una gran 
variedad de configuraciones del modelo utilizando *fuzzy testing*. Para ejecutar las pruebas unitarias solo hace falta ejecutar el comando:

```sh
make test
```

Se estudian otras propiedades del sistema utilizando jupyter notebooks los cuales se 
pueden encontrar en la carpeta [notebooks](./notebooks/).

## Licencia

La licencia del proyecto se puede encontrar en [LICENSE.txt](LICENSE.txt).
