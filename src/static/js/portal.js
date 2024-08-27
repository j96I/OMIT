function previewImage(event) {
  const reader = new FileReader();
  const output = document.createElement('img');
  const tagDisplay = document.querySelector('#tag-display')
  const imagePreview = document.querySelector('#image-preview')
  
  tagDisplay.innerHTML = '';
  imagePreview.innerHTML = '';

  reader.onload = () => {
    output.src = reader.result;
    output.style.maxWidth = '375px';
    output.style.maxHeight = '375px';
    imagePreview.appendChild(output);
  };
  reader.readAsDataURL(event.target.files[0]);
}