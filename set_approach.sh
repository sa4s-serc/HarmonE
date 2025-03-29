#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <baseline>"
  echo "Possible baselines:"
  echo "  Dynamic: harmone, switch, switch+retrain"
  echo "  Single Model: single-lstm, single-svm, single-linear"
  echo "  Single Model with Retraining: single-lstm+retrain, single-svm+retrain, single-linear+retrain"
  exit 1
fi

approach="$1"
echo "$approach" > approach.conf

case $approach in
    harmone|switch|switch+retrain)
        echo "Approach set to '$approach'. No changes to knowledge/model.csv needed."
        ;;
    single-lstm)
        echo "lstm" > knowledge/model.csv
        echo "Approach set to '$approach'. Updated knowledge/model.csv to use 'lstm'."
        ;;
    single-svm)
        echo "svm" > knowledge/model.csv
        echo "Approach set to '$approach'. Updated knowledge/model.csv to use 'svm'."
        ;;
    single-linear)
        echo "linear" > knowledge/model.csv
        echo "Approach set to '$approach'. Updated knowledge/model.csv to use 'linear'."
        ;;
    single-lstm+retrain)
        echo "lstm" > knowledge/model.csv
        echo "Approach set to '$approach'. Updated knowledge/model.csv to use 'lstm'."
        ;;
    single-svm+retrain)
        echo "svm" > knowledge/model.csv
        echo "Approach set to '$approach'. Updated knowledge/model.csv to use 'svm'."
        ;;
    single-linear+retrain)
        echo "linear" > knowledge/model.csv
        echo "Approach set to '$approach'. Updated knowledge/model.csv to use 'linear'."
        ;;
    *)
        echo "Unknown approach option: '$approach'"
        exit 1
        ;;
esac
