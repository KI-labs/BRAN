#!/usr/bin/python
# -*- coding: utf-8 -*-

import click

from .encode import encode_faces
from .detect import face_detection


@click.group()
def bran():
    pass


@bran.command(name='detect')
@click.option("-c", "--cascade", required=True,
              help="path to where the face cascade resides")
@click.option("-e", "--encodings", required=True,
              help="path to serialized db of facial encodings")
@click.option("-p", "--shape_predictor", required=True,
              help="path to facial landmark predictor")
@click.option("-m", "--model_name", required=True, type=click.Choice(['dist_vote', 'dist_avg', 'custom']),
              help="the models to use for face detection")
@click.option("-t", "--confidence", required=False, type=float,
              help="the confidence threshold", default=0.4)
@click.option("-k", "--path_model", required=False, type=str,
              help="the path to the model")
@click.option("-l", "--type", required=False, type=click.Choice(['smile', 'blink']), default='smile',
              help="the liveliness pattern")
@click.option("-r", "--pattern", required=False, type=str, default='ox',
              help="the liveness type, x is a liveliness check and o is not")
@click.option("-f/-no-f", required=False, default=False,
              help="whether to run on pi_camera")
def detection(cascade, encodings, shape_predictor, model_name, confidence, path_model, type, pattern, f):
    face_detection.detect(cascade,
                          encodings,
                          shape_predictor,
                          model_name,
                          confidence,
                          path_model,
                          f,
                          type,
                          pattern)


@bran.command(name='encode')
@click.option("-i", "--dataset", required=True, help="path to input directory of faces + images")
@click.option("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
@click.option("-d", "--detection-method", type=str, default="cnn",
              help="face detection model to use: either 'hog' or 'cnn'")
def encoding(dataset, detection_method, encodings):
    encode_faces.encode(dataset, detection_method, encodings)


def start():
    bran(obj={})


if __name__ == "__main__":
    start()
