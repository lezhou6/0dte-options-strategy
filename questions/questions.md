# DEX correctness

Net DEX on 2026-5-15 looks very weird (large in magnitude and noisy), see netdex20260515.JPG. Potential reasons:  
- 2026-5-15 is a Friday (weekly expiration) and also the third Friday in May (monthly expiration)  
- Bad data? Close to 0 data?  
- Incorrect calculation?  

# Gamma flip

How to formulate Gamma flip?  

Price level where aggregate dealer gamma exposure crosses zero (https://flashalpha.com/concepts/gamma-flip)  
- Follow Flash Alpha may be reasonable and conventional  
- But only one gamma flip data per day  

Strike where cumulative GEX crosses zero at each timestamp
- Different from Flash Alpha definition, less reliable?  
- One data per timestamp

I explored the second formulation in notebooks/06_gamma_flip.ipynb. I haven't decided how to formulate gamma flip yet.  

Once I decide, I can add spot vs flip (underlying price - gamma flip) as a feature too.  