import { describe, expect, it } from 'vitest'

import { normalizeIndexLineEndings } from './vite.config'

describe('hostable index output', () => {
  it('normalizes Windows and mixed line endings before writing static HTML', () => {
    expect(normalizeIndexLineEndings('<head>\r\n</head>\n<body>\r</body>')).toBe(
      '<head>\n</head>\n<body>\n</body>',
    )
  })
})
