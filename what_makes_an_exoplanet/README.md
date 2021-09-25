# **What Makes an Exoplanet?**

Identification of objects of interest as planets using Pandas and Numpy feature analysis and SciPy Pearson correlation and T-tests between candidate and confirmed planet distributions.

**Overview:** A project to identify candidate planets using the distribution from already identified planets in the Kepler K2 survey. The data for both candidate and confirmed planets are organized and cleaned with Pandas and Numpy, then visualized with Matplotlib. The distributions of both are compared and examined with Pearson correlations and T-tests to see if there is a pattern within them which can be used to identify potential planets.

**Results:** While there were quite a few large outliers, interestingly, the majority of candidate planets are faster and larger than already confirmed planets. This would theoretically would make them easier to identify than already identified planets.

Powerpoint presentation

## Background and Motivation

Space is considered the final frontier and with an increase in the interest of space exploration today, more methods for analyzing and examining the cosmos should be created in anticipation for future missions.

<img src="https://i.loli.net/2021/09/26/eXdqP5uMIJbUynW.png" alt="image-20210925160022995" style="zoom:80%;" />

**What is an exoplanet?** 

Any planet orbiting a star outside of our solar system.

**How do scientists detect potential exoplanets?**

<u>Radial Velocity</u>

Gravity of the planet pulling on the star that it orbits causes the star to wobble a bit as the planet travels around the star

<u>Transit</u>

Planet blocks a bit of the light from the star as it travels between the star and our telescope, making the star appear a bit dimmer.

Both methods favor larger planets since this would make the phenomenon larger and easier to detect, but can also occur due to other stellar bodies or defects in the detection equipment.

**How do scientists confirm an object as an exoplanet?**

They need multiple transits of the object of interest to establish a regular pattern and provide more data. This favors faster objects or planets with a shorter year.

Can get confirmations with multiple detections methods from multiple detection equipment.

**Data used in project**

Data used is downloaded from the exoplanet archive from Caltech (link in references). The data was gathered from the Kepler Space Telescope K2 extended mission from 2014-2018 and includes 2582 objects. 371 which were confirmed as planets and 1974 potential planets.

<img src="https://i.loli.net/2021/09/26/EeOjNqsAx1XgJt4.png" alt="image-20210925160202210" style="zoom: 80%;" />

An artist's rendition of the Kepler Space Telescope

<img src="https://i.loli.net/2021/09/26/lxTL1XONuqiawbh.png" alt="image-20210925160103984" style="zoom:80%;" />

Example of one image that the Kepler Space Telescope took with two identified planets highlighted.

## Method

**Pandas DataFrame organization**

Began by first dividing the downloaded file into confirmed and candidate planets. After looking at the long and confusing list of features, I decided on using only two features for my analysis, planet orbital period (year length or speed) and radius (size). These were chosen since they could be understood the easiest with the background information and are directly linked to the observation metrics described. They were also the columns with the fewest number of NaN values.

**Matplotlib data visualization**

The organized DataFrames were then listed from minimum to maximum and plotted were I noticed that there were massive outliers that would heavily skew the data. To counter that, I limited the orbital period to one Earth year and the radius to 11.2 Earth radii, or the size of Jupiter. This makes the features easier to quantify and cuts out outliers. Finally, more visualizations were created to demonstrate the filtered distributions and compare the candidate and confirmed planet distributions.

**SciPy statistical analysis**

These distributions were then compared using a Pearson correlation and a T-test between both with a random selection of 300 planets from both distributions. This was to encompass the majority of the confirmed planets and an equal number of candidate planets.

## Results

### Outlier Example

<img src="https://i.loli.net/2021/09/26/bgP9AnkiCo8xjFY.png" alt="image-20210925153646331" style="zoom:80%;" />

<img src="https://i.loli.net/2021/09/26/wKWXLZm3GSiBDU8.png" alt="image-20210925153728992" style="zoom: 67%;" />

Massive number of outliers that skewed the mean of the distribution heavily and were then cut out. Some notable mentions are a potential planet with an 83,830 Earth day year and a potential planet that has a radius 1,080 times larger than Earth.

### Sampled Planet Years

![image-20210925154000119](https://i.loli.net/2021/09/26/IlyPimwexft9dTW.png)

![image-20210925154128889](https://i.loli.net/2021/09/26/phMPJUW51YqN4oL.png)

Visualizations comparing a sample distribution of both candidate and confirmed planet year length. In the second graph, the mean year length of candidate planets are slightly lower than the mean year length of confirmed planets.

### Sampled Planet Radii

![image-20210925154202198](https://i.loli.net/2021/09/26/8WtlXHG9mrS16n5.png)

![image-20210925154214228](https://i.loli.net/2021/09/26/E4I6NDcWfUL2gPV.png)

Visualizations comparing a sample distribution of both candidate and confirmed planet radii. In the second graph, the mean radii of candidate planets are slightly lower than the mean radii of confirmed planets.

### Patters and statistical analysis

**Pearson correlation between candidate and confirmed planet distributions:**

Year length: 0.987 with a p-value of 1.887e-240

Radius: 0.955 with a p-value of 9.990e-159

Very highly correlated with each other.

**T-test between candidate and confirmed planet distributions:**

Year length: statistic = 0.447 with a p-value of 0.655

Radius: statistic = -3.542 with a p-value of 0.000

Failed to reject the null hypothesis for year length, successfully rejected the null hypothesis for radius.

### Sampled planet radius vs year length 

![image-20210925154553291](C:\Users\seant\AppData\Roaming\Typora\typora-user-images\image-20210925154553291.png)

Filter planets by both year length and radius and compare both. 

Does a larger planet have longer orbits or are further from the star?

**Correlation:** 0.15 with a p-value of 0.000 - very slightly positively correlated

### Conclusion

**It is quite difficult to correctly identify planets from possible candidates.**

There are many outliers which skew data.

Unsure if the candidate and confirmed planets have the same distributions.

Planet size and speed not highly correlated.

Theoretically easier to confirm candidate planets than already confirmed planets based on the means of their distributions.

## References

**Data:**

https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2candidates

https://exoplanets.nasa.gov/what-is-an-exoplanet/in-depth/#otp_how_do_we_find_exoplanets

https://en.wikipedia.org/wiki/Kepler_space_telescope#Finding_planet_candidates