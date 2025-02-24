const arialFontButton = document.getElementById('arialFontButton');
const sansSerifFontButton = document.getElementById('sansSerifFontButton');
const helveticaFontButton = document.getElementById('helveticaFontButton');
const timesRomanFontButton = document.getElementById('timesRomanFontButton');
const fontChangeButton = document.getElementById('fontChangeButton');
const fontSelectionMenu = document.getElementById('fontTypeSelectionContainer');

fontChangeButton.addEventListener('click', () => {
    if (fontSelectionMenu.style.display === 'none' || fontSelectionMenu.style.display === '') {
        fontSelectionMenu.style.display = 'flex';
    } else {
        fontSelectionMenu.style.display = 'none';
    }
});

let contentEdit = document.getElementById('content');

arialFontButton.addEventListener('click', changeFontToArial());

function changeFontToArial() {
    if (!contentEdit.style.fontFamily === 'Arial' || !contentEdit.style.fontFamily === '' || contentEdit.style.fontFamily === '') {
        contentEdit.style.fontFamily = 'Arial';
    }
}

sansSerifFontButton.addEventListener('click', changeFontToSansSerif());

function changeFontToSansSerif() {
    if (!contentEdit.style.fontFamily === 'Sans Serif' || !contentEdit.style.fontFamily === '' || contentEdit.style.fontFamily === '') {
        contentEdit.style.fontFamily = 'Sans Serif';
    }
}
