function interpolateColor(score) {

    parseFloat(score)
    const green = { r: 85, g: 107, b: 47 }
    const red = { r: 238, g: 66, b: 67}

    const r = Math.round((red.r - green.r) * score + green.r)
    const g = Math.round((red.g - green.g) * score + green.g)
    const b = Math.round((red.b - green.b) * score + green.b)

    return `rgb(${r}, ${g}, ${b})`
}

console.log(interpolateColor(0.019877424463629723))
