from traditional_bow.preprocessing import preprocess
from traditional_bow.bow_extractor import BoWExtractor
from core.file_utils import read_raw_text
from pprint import pprint

def main():

    text = read_raw_text("data/example.txt")
    
    bags_of_words = {
    "definition": ["black hole", "object", "dense", "gravity", "mass", "compact", "event horizon", "no escape"],
    "origin": ["einstein", "relativity", "schwarzschild", "oppenheimer", "snyder", "laplace", "michell", "history"],
    "formation": ["collapse", "supernova", "neutron star", "massive", "life cycle", "form", "stellar"],
    "growth": ["accretion", "merge", "absorb", "surroundings", "gas cloud", "solar masses"],
    "detection": ["radiation", "light", "accretion disk", "orbit", "quasar", "visible", "x-ray", "observation"],
    "astrophysical evidence": ["cygnus x-1", "gw170817", "sagittarius a*", "binary", "galaxy", "milky way"],
    "quantum effects": ["hawking radiation", "temperature", "quantum", "spectrum", "curved spacetime"],
    "limits and theories": ["chandrasekhar limit", "tov limit", "singularity", "frozen star", "coordinate", "collapse"],
    "notable scientists": ["einstein", "chandrasekhar", "eddington", "landau", "finkelstein", "droste", "snyder"],
    "consequences": ["no escape", "time dilation", "spacetime", "event horizon", "invisible", "red shift"],
    }

    extractor = BoWExtractor()
    results = extractor.extract(text, bags_of_words) 
    pprint(results)

if __name__ == "__main__":
    main()
