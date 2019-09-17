# run like mvn clean install -DskipTests; ./run.sh dev-vecs.csv

/usr/local/flink-1.8.1/bin/start-cluster.sh; \
/usr/local/flink-1.8.1/bin/flink run \
-c pl.roqad.flinkknn.KNNEvaluator \
./flink-knn/target/flink-knn-0.1.jar \
--input flink-knn/src/main/resources/$1 \
--det ./flink-knn/flink-knn/src/main/resources/det-pairs.csv \
--k 3
--approximate true; \
/usr/local/flink-1.8.1/bin/stop-cluster.sh
