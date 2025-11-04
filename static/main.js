document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('upload-form')
  const busy = document.getElementById('busy')
  if (form && busy) {
    form.addEventListener('submit', () => {
      busy.classList.remove('hidden')
    })
  }
})
