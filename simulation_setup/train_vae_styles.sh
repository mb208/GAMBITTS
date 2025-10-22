#!/bin/sh

folder=data/simulations/2024_12_30/
bash  train_vae_gl.sh train_vae.sh optimism $folder  optimism_interventions_2000.csv
bash  train_vae_gl.sh train_vae.sh formality $folder formality_interventions_2000.csv
bash  train_vae_gl.sh train_vae.sh encouragement $folder encouragement_interventions_2000.csv
bash  train_vae_gl.sh train_vae.sh severity $folder severity_interventions_2000.csv
bash  train_vae_gl.sh train_vae.sh clarity $folder   clarity_interventions_2000.csv
bash  train_vae_gl.sh train_vae.sh humor $folder humor_interventions_2000.csv
bash  train_vae_gl.sh train_vae.sh complexity $folder complexity_interventions_2000.csv
bash  train_vae_gl.sh train_vae.sh vision $folder vision_interventions_2000.csv
bash  train_vae_gl.sh train_vae.sh detail $folder detail_interventions_2000.csv
bash  train_vae_gl.sh train_vae.sh threat $folder threat_interventions_2000.csv
bash  train_vae_gl.sh train_vae.sh urgency $folder urgency_interventions_2000.csv
bash  train_vae_gl.sh train_vae.sh politeness $folder politeness_interventions_2000.csv
bash  train_vae_gl.sh train_vae.sh personalization $folder personalization_interventions_2000.csv 
bash  train_vae_gl.sh train_vae.sh conciseness $folder conciseness_interventions_2000.csv
bash  train_vae_gl.sh train_vae.sh actionability $folder actionability_interventions_2000.csv
bash  train_vae_gl.sh train_vae.sh emotiveness $folder emotiveness_interventions_2000.csv
bash  train_vae_gl.sh train_vae.sh authoritativeness $folder authoritativeness_interventions_2000.csv
bash  train_vae_gl.sh train_vae.sh authenticity $folder authenticity_interventions_2000.csv
bash  train_vae_gl.sh train_vae.sh supportiveness $folder supportiveness_interventions_2000.csv
bash  train_vae_gl.sh train_vae.sh female-codedness $folder female-codedness_interventions_2000.csv
