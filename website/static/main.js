let list = document.querySelectorAll(".navigation li");

function activeLink() {
  list.forEach((item) => {
    item.classList.remove("hovered");
  });
  this.classList.add("hovered"); // Corrected method call
}

// Adding the event listener to each list item
list.forEach((item) => {
  item.addEventListener("mouseover", activeLink); // Corrected function reference
});

let toggle = document.querySelector(".toggle");
let navigation = document.querySelector(".navigation"); // Corrected selector
let main = document.querySelector(".main"); // Corrected selector

toggle.onclick = function () {
  navigation.classList.toggle("active");
  main.classList.toggle("active");
};
