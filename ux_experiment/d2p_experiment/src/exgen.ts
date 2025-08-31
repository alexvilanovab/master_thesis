type DistanceCategory = 'small' | 'medium' | 'large'

type DescriptorPair = {
  distanceCategory: DistanceCategory
  descriptorsMoved: number[]
  a: number[]
  b: number[]
}

const NUM_DESCRIPTORS = 5
const MAX_ATTEMPTS = 100

const MAX_DISTANCE = Math.sqrt(2)
const DISTANCE_MAP: Record<DistanceCategory, number> = {
  small: MAX_DISTANCE * 0.25,
  medium: MAX_DISTANCE * 0.5,
  large: MAX_DISTANCE * 0.75,
}

function randomFloat(): number {
  return Math.random()
}

function generateRandomPoint(): number[] {
  const a = Array.from({ length: NUM_DESCRIPTORS }, randomFloat)
  // ensure that the center is always greater than the start
  if (a[1] > a[2]) [a[1], a[2]] = [a[2], a[1]]
  return a
}

function randomTwoIndices(n: number): number[] {
  const indices = [...Array(n).keys()]
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[indices[i], indices[j]] = [indices[j], indices[i]]
  }
  return indices.slice(0, 2)
}

function randomUnitVector2D(): [number, number] {
  const x = Math.random() * 2 - 1
  const y = Math.random() * 2 - 1
  const norm = Math.sqrt(x * x + y * y)
  return [x / norm, y / norm]
}

function generateB(a: number[], moved: number[], distance: number): number[] | null {
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    const [dx, dy] = randomUnitVector2D()
    const delta = [dx * distance, dy * distance]

    const b = [...a]
    let valid = true

    for (let i = 0; i < moved.length; i++) {
      const idx = moved[i]
      const newVal = b[idx] + delta[i]
      if (newVal < 0 || newVal > 1) {
        valid = false
        break
      }
      b[idx] = newVal
    }

    if (valid && b[1] <= b[2]) return b
  }

  return null
}

function runExperimentFlexible(num: number, allowedCategories: DistanceCategory[]): DescriptorPair[] {
  const results: DescriptorPair[] = []
  while (results.length < num) {
    const category = allowedCategories[Math.floor(Math.random() * allowedCategories.length)]
    let a: number[] = []
    let b: number[] | null = null
    let moved: number[] = []
    while (!b) {
      a = generateRandomPoint()
      moved = randomTwoIndices(NUM_DESCRIPTORS)
      b = generateB(a, moved, DISTANCE_MAP[category])
    }
    results.push({
      distanceCategory: category,
      descriptorsMoved: moved,
      a,
      b,
    })
  }
  return results
}

export function generateExerciseDescriptors(
  num: number,
  allowedCategories?: DistanceCategory[],
): { a: number[]; b: number[] }[] {
  const categoriesToUse = allowedCategories ?? ['small', 'medium', 'large']
  const pairs = runExperimentFlexible(num, categoriesToUse)
  return pairs.map(({ a, b }) => ({
    a: a.map((v) => Math.round(v * 127)),
    b: b.map((v) => Math.round(v * 127)),
  }))
}
