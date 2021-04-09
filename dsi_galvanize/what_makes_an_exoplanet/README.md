# **Identifying Exoplanets**

## **What is an exoplanet?**
Planet orbiting another star than our Sun

## **How do you detect exoplanets?**
Two main methods:

1. Radial Velocity - Gravity of the planet pulling on the star that it orbits star causes the star to wobble a bit as the planet travels around the star

2. Transit - Planet blocks a bit of the light from the star as it travels between our telescope and the star, making the star appear a bit dimmer

- *Both of these methods favor larger planets (planets with larger radii)*

Both of these could also happen due to other stellar bodies or defects in our telescope

## **How do you confirm an object as an exoplanet?**
1.  Need multiple transits of the object in question - provides a regular pattern and more data to confirm the object as an exoplanet

- *Favors faster planets (planets with shorter years)*

2.  Need confirmation from multiple telescopes or with multiple detection methods

https://exoplanets.nasa.gov/what-is-an-exoplanet/in-depth/#otp_how_do_we_find_exoplanets?

## **K2 Survey Data**
Survey done by the Keplar space telescope in its extended mission from 2014-2018

Table includes 2582 objects
- 371 confirmed planets
- 1974 potential planets

https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2candidates

https://en.wikipedia.org/wiki/Kepler_space_telescope#Finding_planet_candidates

## **Outliers**

83,830 day year

1,080 Earth radii

![](images\Keplar_space_image.png =500x500)


Current Data:

Quite a few outliers in planet size and year length

    Much larger than mean

    All are candidates

    No correlation between large planet radius and long year length (Use scipy to prove)

No real correlation between planet size and year length (as far as I can tell, may need to do more work)

Comparison between sampled confirmed and candidate planets show:

    ~same year length, slightly lower year length for candidate planets
    Consisitantly larger candidate planets

    Does not make much sense since larger planets with shorter year lengths would be easier to detect and confirm


on sample?

Use median instead of mean since there are a lot of outliers - skews data
Find correlation between planet size and speed in sample (use scipy)