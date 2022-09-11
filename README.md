# product_classification



---
**NOTE**

In order to run and explore this project you must download all files and place it into **data/** directory.

---
## requirements

This project could be run with:
- Your laptop environment
- Docker container
### Docker

Install Docker following official documentation https://docs.docker.com/get-docker/

### Laptop environment

You need to install:
- Python >= 3.9 https://www.python.org/downloads/
- Poetry https://python-poetry.org/docs/#installation
_____

## Installation
After having followed **requirements** section you are ready to install the project.
In order to develop and run this project you must follow these steps.
Check relevant section according to your need, i.e if you use your **laptop environment** or **docker**
### Laptop environment

1. Clone the project on your laptop using:
```bash
git clone git@github.com:BarthelemyPavy/product_classification.git
```
2. Create a virtual environment, to do so make sure to be at the root directory of the project (same level than pyproject.toml) and run:
```bash
poetry install
source .venv/bin/activate
```
3. Download nltk resources
```bash
poetry run text_resources
```
or
```bash
./post_install.sh
```

### Docker
1. Clone the project on your laptop using:
```bash
git clone git@github.com:BarthelemyPavy/product_classification.git
```
2. Build Docker Image
```bash
docker build -f Dockerfile -t product_classification .
```
Or
```bash
sudo docker build -f Dockerfile -t product_classification .
```
Depend on your install.

3. Docker run
```bash
docker run -it --name product_classification -p 8085:8085 --mount type=bind,src=$(pwd),dst=/home/user/product_classification/ --entrypoint bash product_classification
```
Or
```bash
sudo docker run -it --name product_classification -p 8085:8085 --mount type=bind,src=$(pwd),dst=/home/user/product_classification/ --entrypoint bash product_classification
```
Now, you should be in a terminal into the container.

4. Install dependencies

Make sure to be at the root directory of the project (same level than pyproject.toml) and run:

```bash
poetry install
source .venv/bin/activate
```
5. Download nltk resources
```bash
poetry run text_resources
```
or
```bash
./post_install.sh
```
___

## Execution

For information about job executions, please check [this page](./doc/source/content/execution.md)

___

## Algorithm

For information about implemented algorithm, please check [this page](./doc/source/content/algorithm.md)

___

## Package Architecture

For information the architecture of the solution, please check [this page](./doc/source/content/flows.md)

___


## Next Steps

For information about the next steps, please check [this page](./doc/source/content/next.md)

___
