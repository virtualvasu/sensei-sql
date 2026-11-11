import { useState } from 'react'
import { Check, ChevronsUpDown, Upload } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Command, CommandEmpty, CommandInput, CommandItem, CommandList } from '@/components/ui/command'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'

interface Props {
  databases: { name: string; upload: boolean }[]
  value: string | null
  onChange: (name: string) => void
}

export function DatabasePicker({ databases, value, onChange }: Props) {
  const [open, setOpen] = useState(false)
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger
        render={
          <Button variant="outline" role="combobox" aria-expanded={open} className="w-full justify-between font-normal">
            <span className="truncate">{value ?? 'Select a database…'}</span>
            <ChevronsUpDown className="size-4 shrink-0 opacity-50" />
          </Button>
        }
      />
      <PopoverContent align="start" className="w-[var(--anchor-width)] min-w-64 p-0">
        <Command>
          <CommandInput placeholder="Search databases…" />
          <CommandList>
            <CommandEmpty>No database found.</CommandEmpty>
            {databases.map((d) => (
              <CommandItem
                key={d.name}
                value={d.name}
                onSelect={() => {
                  onChange(d.name)
                  setOpen(false)
                }}
              >
                <Check className={cn('size-4', value === d.name ? 'opacity-100' : 'opacity-0')} />
                <span className="truncate">{d.name}</span>
                {d.upload && (
                  <Badge variant="secondary" className="ml-auto">
                    <Upload className="size-3" /> upload
                  </Badge>
                )}
              </CommandItem>
            ))}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}
