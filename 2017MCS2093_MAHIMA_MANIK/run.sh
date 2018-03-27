# sh run.sh 1 1 q1/imdb/imdb_test_text.txt q1/imdb/imdb_test_labels.txt target
# sh run.sh 1 2 q1/imdb/stemmed_test_text.txt imdb/imdb_test_labels.txt target2
if [ $1 == "1" ]
    then
        if [ $2 == "1" ]
            then python q1/ta_q1a.py q1/modelq1a.p $3 $4 $5
        elif [ $2 == "2" ]
            then python q1/ta_q1d.py q1/modelq1d.p $3 $4 $5
        elif [ $2 == "3" ]
            then python q1/ta_q1d.py q1/modelq1d.p $3 $4 $5
        fi

# sh run.sh 2 1 q2/mnist/sample_mnist_features.csv q2/mnist/sample_mnist_labels.csv target1
# sh run.sh 2 2 /Users/apple/Downloads/libsvm-3.22/python/mnist/sample_mnist_features.csv /Users/apple/Downloads/libsvm-3.22/python/mnist/sample_mnist_labels.csv target2c
# sh run.sh 2 3 /Users/apple/Downloads/libsvm-3.22/python/mnist/sample_mnist_features.csv /Users/apple/Downloads/libsvm-3.22/python/mnist/sample_mnist_labels.csv target2e
elif [ $1 == "2" ]
    then
        if [ $2 == "1" ]
            then python q2/ta_q2b.py q2/modelq2b.p $3 $4 $5
        elif [ $2 == "2" ]
            then python /Users/apple/Downloads/libsvm-3.22/python/ta_q2c.py /Users/apple/Downloads/libsvm-3.22/python/linearmodelq2c $3 $4 $5
        elif [ $2 == "3" ]
            then python /Users/apple/Downloads/libsvm-3.22/python/ta_q2e.py /Users/apple/Downloads/libsvm-3.22/python/modelq2e $3 $4 $5
        fi
fi
