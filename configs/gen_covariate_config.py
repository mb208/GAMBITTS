import os
import yaml

covariate_names = ["optimism", "formality", "severity", "clarity", 
"encouragement", "actionability", "complexity", "supportiveness", "authenticity",
"conciseness", "female-codedness", "personalization", "threat",
 "authoritativeness", "detail","politeness" ,"urgency", "emotiveness", "humor", "vision"]
config_dir = "agent/covariates"

os.makedirs(config_dir, exist_ok=True)

for name in covariate_names:
    path = os.path.join(config_dir, f"{name}.yaml")
    content = {
        name: {
            "text_based": True,
            "one_hot": False
        }
    }
    with open(path, "w") as f:
        yaml.dump(content, f)

print(f"Created {len(covariate_names)} config files in {config_dir}")