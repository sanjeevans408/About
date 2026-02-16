 <!DOCTYPE html>

<html>

<head>

<m  eta charset="UTF-8">

<title>Computer Science Department</title>

<style>

/* Reset some default styles for consistency */ body, h1, h2, p {

margin: 0;

padding: 0;

}



/* Global styles */ body {

font-family: Arial, sans-serif; background-color: #f0f0f0; margin: 0;

padding: 0;

}

 











/* Header styles */ header {

background-color: #17252A; color: #fff;

padding: 20px; text-align: center;

}



header h1 {

font-size: 36px;

}



header p {

font-size: 18px;

}



/* Navigation styles */ nav {

background-color: #3AAFA9; color: #fff;

text-align: center;

}



nav a {

text-decoration: none; color: #fff;

margin: 0 10px;

}



/* Container styles */

.container {

max-width: 1200px; margin: 20px auto; padding: 20px; background-color: #fff;

box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);

}



/* Banner styles */

.banner { display: flex;

justify-content: space-between;

 









align-items: center; padding: 5px;

background-color: #f7f7f7;

}



.banner img {

max-width: 75%;

height: 5%;

}



/* Quick links styles */

.quick-links { margin-top: 20px; text-align: center;

}



.quick-links a { display: inline-block; text-decoration: none; color: #333;

margin: 10px; padding: 10px 20px; border: 1px solid #333; border-radius: 5px;

}



/* Table styles */ table {

width: 100%;

border-collapse: collapse; margin-top: 20px;

}



table, th, td {

border: 1px solid #ccc;

}



th, td {

padding: 10px; text-align: left;

}

</style>

</head>

 









<body>

<header>

<h1></h1>

<h1></h1>

<h1></h1>

<h1 style="text-align: left;"></h1>

<h1 style="text-align: left;"></h1>

<h1 style="text-align: left;"><br>

</h1>

<h1 style="text-align: center;"><img src="file:///C:/Users/gnana/OneDrive/Desktop/logo.jpg"



alt="" style="width: 151px; height: 137px;"></h1>

<h1>Computer Science Department</h1>

<h1></h1>

<h1 style="text-align: left;"></h1>

<p>Welcome to the Computer Science Department at TPGIT</p>

<h1 style="text-align: left;"></h1>

</header>

<nav>

<h1></h1>

<a href="#">Home</a> <a href="#">About Us</a> <a href="#">Courses</a> <a



href="#">Faculty</a> <a href="#">Contact</a> </nav>

<div class="container">

<div class="quick-links">

<h1 style="text-align: left;"></h1>

<a href="#">Explore</a> <a href="#">Learn More</a> <a href="#">Get Started</a> </div>

<h2>Courses Offered</h2>

<table>

<tbody>

<tr>

<th>Course Name</th>

<th>Course Code</th>

<th>Instructor</th>

</tr>

<tr>

<td>Multimedia and Animation</td>

<td>CS5101</td>

<td>Prof. Smith</td>

</tr>

<tr>

 









<td>Web Development</td>

<td>CSCI202</td>

<td>Prof. Johnson</td>

</tr>

</tbody>

</table>

</div>

</body>

</html>



OUTPUT:















RESULT:

Thus, the program to design simple Homepage with Banners, logos, tables quick links, etc is created and the output is verified successfully.

 



 

Exercise No. 4b

 



SEARCH INTERFACE AND SIMPLE NAVIGATION USING BLUEGRIFFON

 









AIM:

To provide a search interface and simple navigation from the home page to the inside pages of the website using BlueGriffon.

ALGORITHM:

a)	Start the program.

b)	Create HTML files for each webpage (index.html, about.html, courses.html, faculty.html, contact.html).

c)	Define the HTML structure :

a)	In each HTML file, define the basic structure with ‘<html>’, ‘<head>’ and ‘<body>’ tags.

b)	Include the necessary metadata within the ‘<head>’ section, such as charset and title.

c)	Add internal CSS styles within ‘<style>’ tag for consistent styling across pages.

d)	Create the header:

a)	Inside the ‘<body>’ of each HTML file, create a ‘<header>’ element to display the page title and brief description.

e)	Create the navigation menu:

a)	Below the header, include a `<nav>` element to create a navigation menu with links to other pages.

b)	Use `<a>` tags for each menu item and set the `href` attribute to link to the respective HTML files.

f)	Create the main content container:

a)	Add a `<div>` element with the class "container" to hold the main content of the page.

b)	Apply styling to this container for a consistent layout.

g)	Populate the content:

a)		Within the container, create a `<div>` with class "content" to hold the specific content of each page.

b)		Add appropriate headings and paragraphs to convey information about the Computer Science Department.

 









c)		For the "Courses" and "Faculty" pages, create tables to list course details and faculty members.

8.	Style the content:

a)	Define CSS styles for various elements to achieve a consistent and visually appealing design.

b)	Customize styles for headers, links, navigation menu, tables, and other page elements.

c)	Use CSS classes and IDs to target specific elements for styling.

9.	Add images:

a)	Include `<img>` tags to display images, such as the department logo and faculty photos.

b)	Set appropriate attributes for image sources, alt text, and dimensions.

10.	Include contact information:

a)	On the "Contact" page, list the department's address, email, and phone number in a structured format.

11.	Testing:

a)	Ensure that all links within the navigation menu correctly point to the corresponding HTML files.

b)	Verify that the website's layout and styling are consistent and visually appealing.

c)	Test the website on different browsers and devices to ensure compatibility.

12.	Additional features (not present in the provided code):

a)	Implement interactive features such as forms for user input or JavaScript functionality.

b)	Add more pages and content as needed for the website's specific requirements.

c)	Consider optimizing the website for search engines (SEO) and improving accessibility.





PROGRAM:

Main code:

<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>Computer Science Department</title>

<style>

 









/* Reset some default styles for consistency */ body, h1, h2, p {

margin: 0;

padding: 0;

}



/* Global styles */ body {

font-family: Arial, sans- serif; background-color: #f0f0f0; margin: 0;

padding: 0;

}



/* Header styles */ header {

background-color: #17252A; color: #fff;

padding: 20px; text-align: center;

}



header h1 {

font-size: 36px;

}



header p {

font-size: 18px;

}



/* Navigation styles */ nav {

background-color: #3AAFA9; color: #fff;

text-align: center;

}



nav a {

text-decoration: none; color: #fff;

margin: 0 10px;

}

 









/* Container styles */

.container {

max-width: 1200px; margin: 20px auto; padding: 20px; background-color: #fff;

box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);

}



/* Banner styles */

.banner { display: flex;

justify-content: space-between; align-items: center;

padding: 5px;

background-color: #f7f7f7;

}



.banner img {

max-width: 75%;

height: 5%;

}



/* Quick links styles */

.quick-links { margin-top: 20px; text-align: center;

}



.quick-links a { display: inline-block; text-decoration: none; color: #333;

margin: 10px; padding: 10px 20px; border: 1px solid #333; border-radius: 5px;

}



/* Table styles */ table {

width: 100%;

border-collapse: collapse;

 









margin-top: 20px;

}



table, th, td {

border: 1px solid #ccc;

}



th, td {

padding: 10px; text-align: left;

}



/* Search bar styles */

.search-container { text-align: center; margin-top: 20px;

}



.search-box { padding: 10px; border: 1px solid #ccc; border-radius: 5px;

}



.search-button { padding: 10px 20px;

background-color: #444; color: #fff;

border: none; border-radius: 5px; cursor: pointer;

}

</style>

</head>

<body>

<header> <img src="file:///C:/Users/gnana/OneDrive/Desktop/logo.jpg" alt="Department Logo"



style="width: 151px; height: 137px;">

<h1>Computer Science Department</h1>

<p>Welcome to the Computer Science Department at TPGIT</p>

</header>

 









<nav> <a href="index.html">Home</a> <a href="C:%5CUsers%5Cgnana%5COneDrive

%5CDesktop%5Cmma%20lab%5CAbout.html">A bout

Us</a> <a href="C:%5CUsers%5Cgnana%5COneDrive%5CDesktop%5Cmma%20lab

%5CCourses.html"> Courses</a>

<a href="C:\Users\gnana\OneDrive\Desktop\mma lab\faculty.html">Faculty</a> <a href="C:\Users\ gnana\OneDrive\Desktop\mma lab\Contact.html">Contact</a> </nav>

<div class="container">

<div class="quick-links"> <a href="#">Explore</a> <a href="#">Learn More</a>

<a href="#">Get Started</a> </div>

<h2>Courses Offered</h2>

<table>

<tbody>

<tr>

<th>Course Name</th>

<th>Course Code</th>

<th>Instructor</th>

</tr>

<tr>

<td>Multimedia and Animation</td>

<td>CS5101</td>

<td>Prof. Smith</td>

</tr>

<tr>

<td>Web Development</td>

<td>CSCI202</td>

<td>Prof. Johnson</td>

</tr>

</tbody>

</table>

</div>

<!-- Search Interface -->

<div class="search-container"> <input class="search-box" placeholder="Search..." type="text"> <button class="search-button">Search</button> </div>

</body>

</html>

 









INDEX.HTML:

<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>Computer Science Department</title>

<style>

/* Reset some default styles for consistency */ body, h1, h2, p {

margin: 0;

padding: 0;

}



/* Global styles */ body {

font-family: Arial, sans- serif; background-color: #f0f0f0; margin: 0;

padding: 0;

}



/* Header styles */ header {

background-color: #17252A; color: #fff;

padding: 20px; text-align: center;

}



header h1 {

font-size: 36px;

}



header p {

font-size: 18px;

}



/* Navigation styles */ nav {

background-color: #3AAFA9; color: #fff;

text-align: center;

}

 











nav a {

text-decoration: none; color: #fff;

margin: 0 10px;

}



/* Container styles */

.container {

max-width: 1200px; margin: 20px auto; padding: 20px; background-color: #fff;

box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);

}



/* Banner styles */

.banner { display: flex;

justify-content: space-between; align-items: center;

padding: 5px;

background-color: #f7f7f7;

}



.banner img {

max-width: 75%;

height: 5%;

}



/* Quick links styles */

.quick-links { margin-top: 20px; text-align: center;

}



.quick-links a { display: inline-block; text-decoration: none; color: #333;

margin: 10px; padding: 10px 20px; border: 1px solid #333;

 









border-radius: 5px;

}



/* Table styles */ table {

width: 100%;

border-collapse: collapse; margin-top: 20px;

}



table, th, td {

border: 1px solid #ccc;

}



th, td {

padding: 10px; text-align: left;

}



/* Search bar styles */

.search-container { text-align: center; margin-top: 20px;

}



.search-box { padding: 10px; border: 1px solid #ccc; border-radius: 5px;

}



.search-button { padding: 10px 20px;

background-color: #444; color: #fff;

border: none; border-radius: 5px; cursor: pointer;

}

</style>

</head>

<body>

 









<header> <img src="file:///C:/Users/gnana/OneDrive/Desktop/logo.jpg" alt="Department Logo"



style="width: 151px; height: 137px;">

<h1>Computer Science Department</h1>

<p>Welcome to the Computer Science Department at TPGIT</p>

</header>

<nav> <a href="C:%5CUsers%5Cgnana%5COneDrive%5CDesktop%5Cmma%20lab

%5Cindex.html">H ome</a>

<a href="about.html">About Us</a> <a href="courses.html">Courses</a> <a



href="faculty.html">Faculty</a> <a href="contact.html">Contact</a> </nav>

<div class="container">

<div class="quick-links"> <a href="#">Explore</a> <a href="#">Learn More</a>

<a href="#">Get Started</a> </div>

<h2>Courses Offered</h2>

<table>

<tbody>

<tr>

<th>Course Name</th>

<th>Course Code</th>

<th>Instructor</th>

</tr>

<tr>

<td>Multimedia and Animation</td>

<td>CS5101</td>

<td>Prof. Smith</td>

</tr>

<tr>

<td>Web Development</td>

<td>CSCI202</td>

<td>Prof. Johnson</td>

</tr>

</tbody>

</table>

</div>

<!-- Search Interface -->

<div class="search-container"> <input class="search-box" placeholder="Search..." type="text"> <button class="search-button">Search</button> </div>

</body>

</html>

 









ABOUT.HTML:

<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>About Us - Computer Science Department</title>

<style>

/* Reset some default styles for consistency */ body, h1, h2, p {

margin: 0;

padding: 0;

}



/* Global styles */ body {

font-family: Arial, sans- serif; background-color: #f0f0f0; margin: 0;

padding: 0;

}



/* Header styles */ header {

background-color: #17252A; color: #fff;

padding: 20px; text-align: center;

}



header h1 {

font-size: 36px;

}



header p {

font-size: 18px;

}



/* Navigation styles */ nav {

background-color: #3AAFA9; color: #fff;

text-align: center;

}

 











nav a {

text-decoration: none; color: #fff;

margin: 0 10px;

}



/* Container styles */

.container {

max-width: 1200px; margin: 20px auto; padding: 20px; background-color: #fff;

box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);

}



/* Content styles */

.content { padding: 20px;

}

</style>

</head>

<body>

<header>

<h1>About Us</h1>

</header>

<nav> <a href="index.html">Home</a> <a href="about.html">About Us</a> <a

href="courses.html">Courses</a> <a href="faculty.html">Faculty</a> <a href="contact.html">Contact</a> </nav>

<div class="container">

<div class="content">

<h2>Welcome to the Computer Science Department</h2>

<p> The Computer Science Department at TPGIT is committed to providing high-quality education and research opportunities in the field of

computer science. Our dedicated faculty members and cutting-edge curriculum ensure that students receive a well-rounded education in computer science. </p>

<p> We offer a wide range of courses, including multimedia and animation, web development, and more. Our experienced instructors are passionate about teaching and mentoring students to help them succeed in their academic and professional careers. </p>

 









</div>

</div>



</body>

</html>



COURSES.HTML:

<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>Computer Science Department</title>

<style>

/* Reset some default styles for consistency */ body, h1, h2, p {

margin: 0;

padding: 0;

}



/* Global styles */ body {

font-family: Arial, sans- serif; background-color: #f0f0f0; margin: 0;

padding: 0;

}



/* Header styles */ header {

background-color: #17252A; color: #fff;

padding: 20px; text-align: center;

}



header h1 {

font-size: 36px;

}



header p {

font-size: 18px;

}

 











/* Navigation styles */ nav {

background-color: #3AAFA9; color: #fff;

text-align: center;

}



nav a {

text-decoration: none; color: #fff;

margin: 0 10px;

}



/* Container styles */

.container {

max-width: 1200px; margin: 20px auto; padding: 20px; background-color: #fff;

box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);

}



/* Banner styles */

.banner { display: flex;

justify-content: space-between; align-items: center;

padding: 5px;

background-color: #f7f7f7;

}



.banner img {

max-width: 75%;

height: 5%;

}



/* Quick links styles */

.quick-links { margin-top: 20px; text-align: center;

}

 









.quick-links a { display: inline-block; text-decoration: none; color: #333;

margin: 10px; padding: 10px 20px; border: 1px solid #333; border-radius: 5px;

}



/* Table styles */ table {

width: 100%;

border-collapse: collapse; margin-top: 20px;

}



table, th, td {

border: 1px solid #ccc;

}



th, td {

padding: 10px; text-align: left;

}



/* Search bar styles */

.search-container { text-align: center; margin-top: 20px;

}



.search-box { padding: 10px; border: 1px solid #ccc; border-radius: 5px;

}



.search-button { padding: 10px 20px;

background-color: #444; color: #fff;

border: none;

 









border-radius: 5px; cursor: pointer;

}

</style>

</head>

<body>

<header> <img src="file:///C:/Users/gnana/OneDrive/Desktop/logo.jpg" alt="Department Logo"



style="width: 151px; height: 137px;">

<h1>Computer Science Department</h1>

<p>Welcome to the Computer Science Department at TPGIT</p>

</header>

<nav> <a href="C:%5CUsers%5Cgnana%5COneDrive%5CDesktop%5Cmma%20lab

%5Cindex.html">H ome</a>

<a href="about.html">About Us</a> <a href="courses.html">Courses</a> <a



href="faculty.html">Faculty</a> <a href="contact.html">Contact</a> </nav>

<div class="container">

<div class="quick-links"> <a href="#">Explore</a> <a href="#">Learn More</a>

<a href="#">Get Started</a> </div>

<h2>Courses Offered</h2>

<table>

<tbody>

<tr>

<th>Course Name</th>

<th>Course Code</th>

<th>Instructor</th>

</tr>

<tr>

<td>Multimedia and Animation</td>

<td>CS5101</td>

<td>Prof. Smith</td>

</tr>

<tr>

<td>Web Development</td>

<td>CSCI202</td>

<td>Prof. Johnson</td>

</tr>

</tbody>

</table>

</div>

 









<!-- Search Interface -->

<div class="search-container"> <input class="search-box" placeholder="Search..." type="text"> <button class="search-button">Search</button> </div>

</body>

</html>



FACULTY.HTML:

<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>Faculty - Computer Science Department</title>

<style>

/* Reset some default styles for consistency */ body, h1, h2, p {

margin: 0;

padding: 0;

}



/* Global styles */ body {

font-family: Arial, sans- serif; background-color: #f0f0f0; margin: 0;

padding: 0;

}



/* Header styles */ header {

background-color: #17252A; color: #fff;

padding: 20px; text-align: center;

}



header h1 {

font-size: 36px;

}



header p {

 









font-size: 18px;

}



/* Navigation styles */ nav {

background-color: #3AAFA9; color: #fff;

text-align: center;

}



nav a {

text-decoration: none; color: #fff;

margin: 0 10px;

}



/* Container styles */

.container {

max-width: 1200px; margin: 20px auto; padding: 20px; background-color: #fff;

box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);

}



/* Content styles */

.content { padding: 20px;

}



/* Faculty list styles */

.faculty-list {

list-style-type: none; margin: 0;

padding: 0;

}



.faculty-item {

margin-bottom: 20px; padding: 10px; border: 1px solid #ccc; border-radius: 5px;

}

 











/* Faculty image styles */

.faculty-image { max-width: 100%; height: auto;

}

</style>

</head>

<body>

<header>

<h1>Faculty</h1>

</header>

<nav> <a href="index.html">Home</a> <a href="about.html">About Us</a> <a href="courses.html">Courses</a> <a href="faculty.html">Faculty</a> <a

href="contact.html">Contact</a> </nav>

<div class="container">

<div class="content">

<h2>Our Faculty</h2>

<ul class="faculty-list">

<li class="faculty-item"> <img class="faculty-image" src="file:///C:/Users/gnana/Downloads/male.jpg"



alt="Prof. John Doe" style="width: 114px; height: 89px;">

<h3>Prof. John Doe</h3>

<p>Teaches Computer Science</p>

</li>

<li class="faculty-item"> <img class="faculty-image" src="file:///C:/Users/gnana/Downloads/male.jpg"



alt="Prof. Jane Smith" style="width: 113px; height: 102px;">

<h3>Prof. Jane Smith</h3>

<p>Teaches Web Development</p>

</li>

<!-- Add more faculty members as needed -->

</ul>

</div>

</div>

</body>

</html>

 









CONTACT.HTML:

<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>Contact - Computer Science Department</title>

<style>

/* Reset some default styles for consistency */ body, h1, h2, p {

margin: 0;

padding: 0;

}



/* Global styles */ body {

font-family: Arial, sans- serif; background-color: #f0f0f0; margin: 0;

padding: 0;

}



/* Header styles */ header {

background-color: #17252A; color: #fff;

padding: 20px; text-align: center;

}



header h1 {

font-size: 36px;

}



header p {

font-size: 18px;

}



/* Navigation styles */ nav {

background-color: #3AAFA9; color: #fff;

text-align: center;

}

 











nav a {

text-decoration: none; color: #fff;

margin: 0 10px;

}



/* Container styles */

.container {

max-width: 1200px; margin: 20px auto; padding: 20px; background-color: #fff;

box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);

}



/* Content styles */

.content { padding: 20px;

}



/* Contact information styles */

.contact-info { margin-top: 20px;

}



.contact-info h3 { font-size: 24px;

margin-bottom: 10px;

}



.contact-info p { font-size: 18px;

}

</style>

</head>

<body>

<header>

<h1>Contact</h1>

</header>

<nav> <a href="index.html">Home</a> <a href="about.html">About Us</a> <a href="courses.html">Courses</a> <a href="faculty.html">Faculty</a> <a

 











href="contact.html">Contact</a> </nav>

<div class="container">

<div class="content">

<h2>Contact Information</h2>

<div class="contact-info">

<h3>Department Address:</h3>

<p>Thanthai Periyar govt institute of technology,</p>

<p>Bagayam,</p>

<p>Vellore-02.</p>

</div>

<div class="contact-info">

<h3>Email:</h3>

<p>csdept@gmail.com</p>

</div>

<div class="contact-info">

<h3>Phone:</h3>

<p>+1 (123) 456-7890</p>

</div>

</div>

</div>

</body>

</html>



OUTPUT:



 









 











 









 




