// present the user with a large preview of the image.
function previewImage(event) {
  const reader = new FileReader();
  const output = document.createElement('img');
  const tagDisplay = document.querySelector('#tag-display')
  const imagePreview = document.querySelector('#image-preview')
  
  tagDisplay.innerHTML = '';
  imagePreview.innerHTML = '';

  reader.onload = () => {
    output.src = reader.result;
    output.style.maxWidth = '30rem';
    output.style.maxHeight = '30rem';
    imagePreview.style.filter = "blur(15px)";
    imagePreview.appendChild(output);
  };
  reader.readAsDataURL(event.target.files[0]);
}

//blur out the image, so the user cannot see any details.
function blurPreview(event) {
  const imagePreview = document.querySelector('#image-preview');
  imagePreview.style.filter = "blur(15px)";
}

// remove the blur effect, so the image is clear.
function showPreview(event) {
  const imagePreview = document.querySelector('#image-preview');
  imagePreview.style.filter = "none";
}

// show the image for a fraction of a second, then blur it again automatically.
function blinkPreview(event) {
  showPreview(event);
  setTimeout(()=>(blurPreview(event)),100);
}

