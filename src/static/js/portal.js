function previewImage(event) {
  const reader = new FileReader();
  const output = document.createElement('img');
  const previewContainer = document.querySelector('#preview-container')
  
  reader.onload = () => {
    output.src = reader.result;
    output.style.maxWidth = '375px';
    output.style.maxHeight = '375px';
    previewContainer.appendChild(output);
  };
  reader.readAsDataURL(event.target.files[0]);
}