import D2P from './D2P'
import { D2PProvider, useD2P } from './D2PContext'
import { generateExerciseDescriptors } from './exgen'
import cn from 'classnames'
import { useState, useEffect } from 'react'
import { v4 as uuidv4 } from 'uuid'

type Survey = {
  id: string
  // Background
  consent: boolean
  ageRange: string
  yearsStudying: string
  yearsPerforming: string
  yearsPercussion: string
  // Exercises
  exerciseResults: {
    [key: number]: {
      patternA: number[]
      patternB: number[]
      descriptorsA: number[]
      descriptorsB: number[]
      initialDescriptorsB: number[]
      subjectiveSimilarity: number
      elapsedTime: number // Time in seconds
    }
  }
  // Feedback
  interfaceIntuition: string
  onsetCountDifficulty: string
  startDifficulty: string
  centerDifficulty: string
  syncopationDifficulty: string
  balanceDifficulty: string
  confusingComment: string
  likedComment: string
  feedbackComment: string
}

const EXERCISE_DESCRIPTORS = generateExerciseDescriptors(8, ['medium'])

type ExerciseProps = {
  currentExercise: number
  showRating: boolean
  setShowRating: (showRating: boolean) => void
  survey: Survey
  setSurvey: (survey: Survey) => void
  onNext: () => void
  startTime: number
}

function Exercise({ currentExercise, showRating, setShowRating, survey, setSurvey, onNext, startTime }: ExerciseProps) {
  const { stopAll } = useD2P()
  const descriptors = EXERCISE_DESCRIPTORS[currentExercise - 1]
  const [patternA, setPatternA] = useState<number[]>([])
  const [patternB, setPatternB] = useState<number[]>([])
  const [descriptorsB, setDescriptorsB] = useState<number[]>(descriptors.b)

  const handleDone = () => {
    stopAll()
    setShowRating(true)
  }

  const handleRating = (value: number) => {
    const elapsedTime = Math.round((Date.now() - startTime) / 1000) // Convert to seconds and round
    setSurvey({
      ...survey,
      exerciseResults: {
        ...survey.exerciseResults,
        [currentExercise]: {
          patternA: patternA.map((v) => Number(v.toFixed(2))),
          patternB: patternB.map((v) => Number(v.toFixed(2))),
          descriptorsA: descriptors.a,
          descriptorsB: descriptorsB,
          initialDescriptorsB: descriptors.b,
          subjectiveSimilarity: value,
          elapsedTime,
        },
      },
    })
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex justify-center">
        <div className="font-bold">{`[${currentExercise}/8]`}</div>
      </div>
      <div className="flex w-full flex-col gap-4">
        <div className="flex gap-2">
          <span className="font-bold">{`#${-1 + currentExercise * 2}`}</span>
          <span>{`Listen to this pattern.`}</span>
        </div>
        <D2P
          key={`d2p_ex${currentExercise}_a`}
          id={`d2p_ex${currentExercise}_a`}
          showDescriptorKnobs={false}
          showControls={false}
          d0={descriptors.a[0]}
          d1={descriptors.a[1]}
          d2={descriptors.a[2]}
          d3={descriptors.a[3]}
          d4={descriptors.a[4]}
          onPatternChange={setPatternA}
        />
      </div>
      <div className="flex w-full flex-col gap-4">
        <div className="flex gap-2">
          <span className="font-bold">{`#${currentExercise * 2}`}</span>
          <div className="flex flex-col gap-1">
            <span>{`Use the sliders below to transform the next pattern into the one you just listened to (#${-1 + currentExercise * 2}).`}</span>
            <span>{`Try to get as close as possible, but don't worry if they're not identical.`}</span>
            <span>{`Click "Done" when you're ready.`}</span>
          </div>
        </div>
        <D2P
          key={`d2p_ex${currentExercise}_b`}
          id={`d2p_ex${currentExercise}_b`}
          showDescriptorKnobs={showRating ? false : true}
          d0={descriptors.b[0]}
          d1={descriptors.b[1]}
          d2={descriptors.b[2]}
          d3={descriptors.b[3]}
          d4={descriptors.b[4]}
          onPatternChange={setPatternB}
          onDescriptorChange={(values) => setDescriptorsB([values.d0, values.d1, values.d2, values.d3, values.d4])}
        />
      </div>
      <div className="flex w-full justify-center gap-4">
        {!showRating ? (
          <button
            className={
              'h-8 cursor-pointer rounded border border-gray-300 bg-gray-50 px-3 text-xs hover:bg-gray-100 disabled:cursor-not-allowed disabled:opacity-50'
            }
            onClick={handleDone}
          >
            Done
          </button>
        ) : (
          <>
            <div className="flex items-center gap-4">
              <span>{`How similar do you think the two patterns are?`}</span>
              <div className="flex gap-2">
                {[0, 1, 2, 3, 4, 5].map((value) => (
                  <div key={value} className="flex items-center gap-1">
                    <input
                      type="radio"
                      id={`similarity-${value}`}
                      name="similarity"
                      value={value}
                      checked={survey.exerciseResults[currentExercise]?.subjectiveSimilarity === value}
                      onChange={() => handleRating(value)}
                      required
                    />
                    <label htmlFor={`similarity-${value}`}>{value}</label>
                  </div>
                ))}
              </div>
            </div>
            <button
              className={
                'h-8 cursor-pointer rounded border border-gray-300 bg-gray-50 px-3 text-xs hover:bg-gray-100 disabled:cursor-not-allowed disabled:opacity-50'
              }
              disabled={survey.exerciseResults[currentExercise]?.subjectiveSimilarity === undefined}
              onClick={onNext}
            >
              {`Next`}
            </button>
          </>
        )}
      </div>
    </div>
  )
}

function AppContent() {
  const [currentStep, setCurrentStep] = useState(0)
  const [currentExercise, setCurrentExercise] = useState(1)
  const [showRating, setShowRating] = useState(false)
  const { stopAll, playedD2Ps } = useD2P()
  const [startTime, setStartTime] = useState(0)

  const [survey, setSurvey] = useState<Survey>({
    id: uuidv4(),
    // Initial form
    consent: false,
    ageRange: '',
    yearsStudying: '',
    yearsPerforming: '',
    yearsPercussion: '',
    // Exercise ratings
    exerciseResults: {},
    // Feedback form
    interfaceIntuition: '',
    onsetCountDifficulty: '',
    startDifficulty: '',
    centerDifficulty: '',
    syncopationDifficulty: '',
    balanceDifficulty: '',
    confusingComment: '',
    likedComment: '',
    feedbackComment: '',
  })

  const handleFormSubmit = (e: React.FormEvent) => e.preventDefault()

  const handleNext = () => {
    stopAll()
    if (currentStep === 1) setStartTime(Date.now()) // Set start time for for 1st exercise
    setCurrentStep((prev) => prev + 1)
  }

  const handleExerciseNext = () => {
    if (currentExercise < 8) {
      stopAll()
      setCurrentExercise((prev) => prev + 1)
      setShowRating(false)
      setStartTime(Date.now()) // Reset start time for next exercise
    } else handleNext()
  }

  const handleFinish = () => handleNext()

  useEffect(() => {
    if (currentStep === 4) {
      // Create CSV header
      const header = [
        // Background
        'id',
        'consent',
        'ageRange',
        'yearsStudying',
        'yearsPerforming',
        'yearsPercussion',
        // Exercise
        'exerciseNumber',
        'patternA',
        'patternB',
        'descriptorsA',
        'descriptorsB',
        'initialDescriptorsB',
        'elapsedTime',
        'subjectiveSimilarity',
        // Feedback
        'interfaceIntuition',
        'onsetCountDifficulty',
        'startDifficulty',
        'centerDifficulty',
        'syncopationDifficulty',
        'balanceDifficulty',
        'confusingComment',
        'likedComment',
        'feedbackComment',
      ].join(',')

      // Create rows (one per exercise)
      const rows = Object.entries(survey.exerciseResults).map(([exerciseNumber, result]) => {
        return [
          // Background (same for all rows)
          survey.id,
          survey.consent,
          survey.ageRange,
          survey.yearsStudying,
          survey.yearsPerforming,
          survey.yearsPercussion,
          // Exercise specific
          exerciseNumber,
          result.patternA.join(' '),
          result.patternB.join(' '),
          result.descriptorsA.join(' '),
          result.descriptorsB.join(' '),
          result.initialDescriptorsB.join(' '),
          result.elapsedTime,
          result.subjectiveSimilarity,
          // Feedback (same for all rows)
          survey.interfaceIntuition,
          survey.onsetCountDifficulty,
          survey.startDifficulty,
          survey.centerDifficulty,
          survey.syncopationDifficulty,
          survey.balanceDifficulty,
          `"${survey.confusingComment.replace(/"/g, '""')}"`,
          `"${survey.likedComment.replace(/"/g, '""')}"`,
          `"${survey.feedbackComment.replace(/"/g, '""')}"`,
        ].join(',')
      })

      // Combine header and rows
      const csv = [header, ...rows].join('\n')

      // Create and download file
      // const blob = new Blob([csv], { type: 'text/csv' })
      // const url = URL.createObjectURL(blob)
      // const a = document.createElement('a')
      // a.href = url
      // a.download = `d2p_experiment_${survey.id}.csv`
      // document.body.appendChild(a)
      // a.click()
      // document.body.removeChild(a)
      // URL.revokeObjectURL(url)

      fetch(
        'https://script.google.com/macros/s/AKfycbz49YiQdiyWsKclshykd5OwgKRz1FUnquOXWvg5so393y0dTuxKEX1AG_CIqtDAy7h0gA/exec',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded', // 👈 Avoids preflight!
          },
          body: new URLSearchParams({ csv }).toString(), // 👈 Send CSV as plain form data
        },
      )
        .then(() => handleNext())
        .catch(() => handleNext())
    }
  }, [
    currentStep,
    survey.id,
    survey.consent,
    survey.ageRange,
    survey.yearsStudying,
    survey.yearsPerforming,
    survey.yearsPercussion,
    survey.interfaceIntuition,
    survey.onsetCountDifficulty,
    survey.startDifficulty,
    survey.centerDifficulty,
    survey.syncopationDifficulty,
    survey.balanceDifficulty,
    survey.confusingComment,
    survey.likedComment,
    survey.feedbackComment,
    survey.exerciseResults,
  ])

  if (currentStep === 5) {
    return (
      <div className="flex h-screen w-screen flex-col items-center justify-center gap-8 bg-green-800 font-mono text-sm transition-opacity duration-300">
        <div className="flex w-full flex-col items-center gap-2 rounded p-4 text-green-100">
          <span>{`Thank you for participating in this experiment!`}</span>
          <span>{`Your data has been saved and you can close this window now.`}</span>
          <span>
            {`Feel free to contact `}
            <a className="font-bold underline" href="mailto:alex@alexvilanova.com">
              {`alex@vilanova.dev`}
            </a>
            {` if you have any questions.`}
          </span>
        </div>
      </div>
    )
  }

  return (
    <div className="mx-auto mt-4 flex max-w-4xl flex-col gap-4 px-4 font-mono text-sm">
      <div className="flex flex-col gap-2 rounded border-4 border-dashed border-yellow-600 bg-yellow-500 p-4 text-yellow-900">
        <span>{`Welcome! In this experiment, you'll explore a rhythm generation tool powered by machine learning. Your task is to learn how it works, complete challenges, and share feedback. The experiment will take around 15 minutes and it is important that you give it your full attention and focus.`}</span>
        <div className="flex flex-col text-xs">
          <span className="font-bold">{`Consent and Data Use Notice`}</span>
          <span>{`
            By participating in this experiment, you agree that your interaction data (e.g., slider values, patterns you create, and timing data) may be anonymously recorded and analyzed for academic research purposes.
            No personally identifiable information is collected.
            All data will be stored securely and used solely for the purpose of understanding user interaction with rhythm generation tools.
            Participation is voluntary, and you may stop at any time by closing the browser window.
          `}</span>
        </div>
        <div className="text-xs">
          <span>{`For questions, contact `}</span>
          <a className="font-bold underline" href="mailto:alex@vilanova.dev">{`alex@vilanova.dev`}</a>
        </div>
      </div>

      <hr className="w-full border border-gray-200" />

      <form
        onSubmit={handleFormSubmit}
        className={cn(
          'flex flex-col gap-4 transition-opacity duration-300',
          currentStep !== 0 ? 'pointer-events-none opacity-40 select-none' : 'opacity-100',
        )}
      >
        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="consent"
            checked={survey.consent}
            onChange={(e) => setSurvey({ ...survey, consent: e.target.checked })}
            required
          />
          <label htmlFor="consent">I agree to participate in this study.</label>
        </div>

        <div className="flex flex-col gap-2">
          <span>{`Age range:`}</span>
          <div className="flex flex-wrap gap-4">
            {[
              { value: '18-25', label: '18-25' },
              { value: '26-35', label: '26-35' },
              { value: '36-45', label: '36-45' },
              { value: '46-55', label: '46-55' },
              { value: '56+', label: '56+' },
            ].map(({ value, label }) => (
              <div key={value} className="flex items-center gap-2">
                <input
                  type="radio"
                  id={`ageRange-${value}`}
                  name="ageRange"
                  value={value}
                  checked={survey.ageRange === value}
                  onChange={(e) => setSurvey({ ...survey, ageRange: e.target.value })}
                  required
                />
                <label htmlFor={`ageRange-${value}`}>{label}</label>
              </div>
            ))}
          </div>
        </div>

        <div className="flex flex-col gap-2">
          <span>{`Number of years spent studying music:`}</span>
          <div className="flex flex-wrap gap-4">
            {[
              { value: '0', label: '0' },
              { value: '1', label: '1' },
              { value: '2', label: '2' },
              { value: '3', label: '3' },
              { value: '4+', label: '4+' },
            ].map(({ value, label }) => (
              <div key={value} className="flex items-center gap-2">
                <input
                  type="radio"
                  id={`yearsStudying-${value}`}
                  name="yearsStudying"
                  value={value}
                  checked={survey.yearsStudying === value}
                  onChange={(e) => setSurvey({ ...survey, yearsStudying: e.target.value })}
                  required
                />
                <label htmlFor={`yearsStudying-${value}`}>{label}</label>
              </div>
            ))}
          </div>
        </div>

        <div className="flex flex-col gap-2">
          <span>{`Number of years spent performing music:`}</span>
          <div className="flex flex-wrap gap-4">
            {[
              { value: '0', label: '0' },
              { value: '1', label: '1' },
              { value: '2', label: '2' },
              { value: '3', label: '3' },
              { value: '4+', label: '4+' },
            ].map(({ value, label }) => (
              <div key={value} className="flex items-center gap-2">
                <input
                  type="radio"
                  id={`yearsPerforming-${value}`}
                  name="yearsPerforming"
                  value={value}
                  checked={survey.yearsPerforming === value}
                  onChange={(e) => setSurvey({ ...survey, yearsPerforming: e.target.value })}
                  required
                />
                <label htmlFor={`yearsPerforming-${value}`}>{label}</label>
              </div>
            ))}
          </div>
        </div>

        <div className="flex flex-col gap-2">
          <span>{`Number of years performing percussion instruments:`}</span>
          <div className="flex flex-wrap gap-4">
            {[
              { value: '0', label: '0' },
              { value: '1', label: '1' },
              { value: '2', label: '2' },
              { value: '3', label: '3' },
              { value: '4+', label: '4+' },
            ].map(({ value, label }) => (
              <div key={value} className="flex items-center gap-2">
                <input
                  type="radio"
                  id={`yearsPercussion-${value}`}
                  name="yearsPercussion"
                  value={value}
                  checked={survey.yearsPercussion === value}
                  onChange={(e) => setSurvey({ ...survey, yearsPercussion: e.target.value })}
                  required
                />
                <label htmlFor={`yearsPercussion-${value}`}>{label}</label>
              </div>
            ))}
          </div>
        </div>

        <div className="flex w-full justify-center">
          <button
            className={
              'h-8 cursor-pointer rounded border border-gray-300 bg-gray-50 px-3 text-xs hover:bg-gray-100 disabled:cursor-not-allowed disabled:opacity-50'
            }
            disabled={
              !survey.consent ||
              !survey.ageRange ||
              !survey.yearsStudying ||
              !survey.yearsPerforming ||
              !survey.yearsPercussion
            }
            onClick={handleNext}
          >
            Next
          </button>
        </div>
      </form>

      <hr className="w-full border border-gray-200" />

      <div
        className={cn(
          'flex w-full flex-col gap-4 transition-opacity duration-300',
          currentStep !== 1 ? 'pointer-events-none opacity-40 select-none' : 'opacity-100',
        )}
      >
        <div>
          <div className="flex gap-2">
            <span className="font-bold">{`#0`}</span>
            <div className="flex flex-col gap-1">
              <span>{`Play around with the sliders, get familiar with the interface.`}</span>
              <span>{`The sliders below control the following parameters:`}</span>
              <ul className="list-inside list-disc">
                <li>
                  <span className="font-bold">{`onset_count`}</span>: {`number of onsets in the resulting pattern.`}
                </li>
                <li>
                  <span className="font-bold">{`start`}</span>: {`position of the first onset of the pattern.`}
                </li>
                <li>
                  <span className="font-bold">{`center`}</span>: {`mass center of the pattern.`}
                </li>
                <li>
                  <span className="font-bold">{`syncopation`}</span>: {`how syncopated are the onsets in the pattern.`}
                </li>
                <li>
                  <span className="font-bold">{`balance`}</span>:{' '}
                  {`how well distributed are the onsets in the pattern.`}
                </li>
              </ul>
            </div>
          </div>
        </div>
        <D2P id="d2p_fam" showDescriptorKnobs={true} showControls={true} />
        <div className="flex w-full justify-center">
          <button
            className={
              'h-8 cursor-pointer rounded border border-gray-300 bg-gray-50 px-3 text-xs hover:bg-gray-100 disabled:cursor-not-allowed disabled:opacity-50'
            }
            disabled={!playedD2Ps.has('d2p_fam')}
            onClick={handleNext}
          >
            Next
          </button>
        </div>
      </div>

      <hr className="w-full border border-gray-200" />

      <div
        className={cn(
          'flex flex-col gap-6 transition-opacity duration-300',
          currentStep !== 2 ? 'pointer-events-none opacity-40 select-none' : 'opacity-100',
        )}
      >
        <Exercise
          currentExercise={currentExercise}
          showRating={showRating}
          setShowRating={setShowRating}
          survey={survey}
          setSurvey={setSurvey}
          onNext={handleExerciseNext}
          startTime={startTime}
        />
      </div>

      <hr className="w-full border border-gray-200" />

      <form
        onSubmit={handleFormSubmit}
        className={cn(
          'flex flex-col gap-4 transition-opacity duration-300',
          currentStep !== 3 ? 'pointer-events-none opacity-40 select-none' : 'opacity-100',
        )}
      >
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-2">
            <div className="flex gap-2">
              <span className="font-bold">{`#${EXERCISE_DESCRIPTORS.length * 2 + 1}`}</span>
              <span>{`How intuitive was the interface?`}</span>
            </div>
            <div className="flex flex-wrap gap-4">
              {[0, 1, 2, 3, 4, 5].map((value) => (
                <div key={value} className="flex items-center gap-2">
                  <input
                    type="radio"
                    id={`interfaceIntuition-${value}`}
                    name="interfaceIntuition"
                    value={value}
                    checked={survey.interfaceIntuition === value.toString()}
                    onChange={(e) => setSurvey({ ...survey, interfaceIntuition: e.target.value })}
                    required
                  />
                  <label htmlFor={`interfaceIntuition-${value}`}>{value}</label>
                </div>
              ))}
            </div>
          </div>

          <div className="flex flex-col gap-2">
            <div className="flex flex-col gap-4">
              <div className="flex flex-col gap-2">
                <div className="flex gap-2">
                  <span className="font-bold">{`#${EXERCISE_DESCRIPTORS.length * 2 + 2}`}</span>
                  <span>{`How difficult was it to use the sliders?`}</span>
                </div>
              </div>
            </div>
            <div className="ml-4">
              <table>
                <tbody>
                  <tr>
                    <td className="w-32">
                      <label htmlFor="onsetCountDifficulty-0" className="font-bold">{`onset_count`}</label>
                    </td>
                    <td>
                      <div className="flex gap-4">
                        {[0, 1, 2, 3, 4, 5].map((value) => (
                          <div key={value} className="flex items-center gap-2">
                            <input
                              type="radio"
                              id={`onsetCountDifficulty-${value}`}
                              name="onsetCountDifficulty"
                              value={value}
                              checked={survey.onsetCountDifficulty === value.toString()}
                              onChange={(e) => setSurvey({ ...survey, onsetCountDifficulty: e.target.value })}
                              required
                            />
                            <label htmlFor={`onsetCountDifficulty-${value}`}>{value}</label>
                          </div>
                        ))}
                      </div>
                    </td>
                  </tr>
                  <tr>
                    <td className="w-32">
                      <label htmlFor="startDifficulty-0" className="font-bold">{`start`}</label>
                    </td>
                    <td>
                      <div className="flex gap-4">
                        {[0, 1, 2, 3, 4, 5].map((value) => (
                          <div key={value} className="flex items-center gap-2">
                            <input
                              type="radio"
                              id={`startDifficulty-${value}`}
                              name="startDifficulty"
                              value={value}
                              checked={survey.startDifficulty === value.toString()}
                              onChange={(e) => setSurvey({ ...survey, startDifficulty: e.target.value })}
                              required
                            />
                            <label htmlFor={`startDifficulty-${value}`}>{value}</label>
                          </div>
                        ))}
                      </div>
                    </td>
                  </tr>
                  <tr>
                    <td className="w-32">
                      <label htmlFor="centerDifficulty-0" className="font-bold">{`center`}</label>
                    </td>
                    <td>
                      <div className="flex gap-4">
                        {[0, 1, 2, 3, 4, 5].map((value) => (
                          <div key={value} className="flex items-center gap-2">
                            <input
                              type="radio"
                              id={`centerDifficulty-${value}`}
                              name="centerDifficulty"
                              value={value}
                              checked={survey.centerDifficulty === value.toString()}
                              onChange={(e) => setSurvey({ ...survey, centerDifficulty: e.target.value })}
                              required
                            />
                            <label htmlFor={`centerDifficulty-${value}`}>{value}</label>
                          </div>
                        ))}
                      </div>
                    </td>
                  </tr>
                  <tr>
                    <td className="w-32">
                      <label htmlFor="syncopationDifficulty-0" className="font-bold">{`syncopation`}</label>
                    </td>
                    <td>
                      <div className="flex gap-4">
                        {[0, 1, 2, 3, 4, 5].map((value) => (
                          <div key={value} className="flex items-center gap-2">
                            <input
                              type="radio"
                              id={`syncopationDifficulty-${value}`}
                              name="syncopationDifficulty"
                              value={value}
                              checked={survey.syncopationDifficulty === value.toString()}
                              onChange={(e) => setSurvey({ ...survey, syncopationDifficulty: e.target.value })}
                              required
                            />
                            <label htmlFor={`syncopationDifficulty-${value}`}>{value}</label>
                          </div>
                        ))}
                      </div>
                    </td>
                  </tr>
                  <tr>
                    <td className="w-32">
                      <label htmlFor="balanceDifficulty-0" className="font-bold">{`balance`}</label>
                    </td>
                    <td>
                      <div className="flex gap-4">
                        {[0, 1, 2, 3, 4, 5].map((value) => (
                          <div key={value} className="flex items-center gap-2">
                            <input
                              type="radio"
                              id={`balanceDifficulty-${value}`}
                              name="balanceDifficulty"
                              value={value}
                              checked={survey.balanceDifficulty === value.toString()}
                              onChange={(e) => setSurvey({ ...survey, balanceDifficulty: e.target.value })}
                              required
                            />
                            <label htmlFor={`balanceDifficulty-${value}`}>{value}</label>
                          </div>
                        ))}
                      </div>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <div className="flex flex-col gap-2">
            <div className="flex gap-2">
              <span className="font-bold">{`#${EXERCISE_DESCRIPTORS.length * 2 + 3}`}</span>
              <label htmlFor="confusingComment">{`What was the most confusing part of the interface?`}</label>
            </div>
            <textarea
              id="confusingComment"
              value={survey.confusingComment}
              onChange={(e) => setSurvey({ ...survey, confusingComment: e.target.value })}
              required
            />
          </div>

          <div className="flex flex-col gap-2">
            <div className="flex gap-2">
              <span className="font-bold">{`#${EXERCISE_DESCRIPTORS.length * 2 + 4}`}</span>
              <label htmlFor="likedComment">{`What did you like about the interface?`}</label>
            </div>
            <textarea
              id="likedComment"
              value={survey.likedComment}
              onChange={(e) => setSurvey({ ...survey, likedComment: e.target.value })}
              required
            />
          </div>

          <div className="flex flex-col gap-2">
            <div className="flex gap-2">
              <span className="font-bold">{`#${EXERCISE_DESCRIPTORS.length * 2 + 5}`}</span>
              <label htmlFor="feedbackComment">{`Do you have any feedback, suggestions, or ideas?`}</label>
            </div>
            <textarea
              id="feedbackComment"
              value={survey.feedbackComment}
              onChange={(e) => setSurvey({ ...survey, feedbackComment: e.target.value })}
              required
            />
          </div>
        </div>

        <div className="flex w-full justify-center">
          <button
            className={
              'h-8 cursor-pointer rounded border border-gray-300 bg-gray-50 px-3 text-xs hover:bg-gray-100 disabled:cursor-not-allowed disabled:opacity-50'
            }
            disabled={
              !survey.interfaceIntuition ||
              !survey.onsetCountDifficulty ||
              !survey.startDifficulty ||
              !survey.centerDifficulty ||
              !survey.syncopationDifficulty ||
              !survey.balanceDifficulty
            }
            onClick={handleFinish}
          >
            {`Finish`}
          </button>
        </div>
      </form>

      <br />
    </div>
  )
}

export default function App() {
  return (
    <D2PProvider>
      <AppContent />
    </D2PProvider>
  )
}
