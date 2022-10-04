source ./.env

docker-compose -f ./docker-compose.yaml run --rm run \
		bash -c "python ./main.py $1 $2"
